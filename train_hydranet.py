import torch
from torch import nn
from detectron2.modeling import build_backbone
from detectron2.modeling import (
    build_proposal_generator,
    build_roi_heads,
    build_sem_seg_head,
    META_ARCH_REGISTRY,
)
from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.panoptic_fpn import sem_seg_postprocess
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from dataset import get_dataset_dicts


class DynamicWeightAveraging:
    def __init__(self, num_tasks, T=2.0, K=None):
        self.num_tasks = num_tasks
        self.T = T
        self.K = K if K is not None else num_tasks
        self.loss_history = torch.ones(num_tasks, 2)

    def update_weights(self, losses):
        wk = self.loss_history[:, 0] / self.loss_history[:, 1]

        exp_wk = torch.exp(wk / self.T)
        weights = self.K * exp_wk / torch.sum(exp_wk)

        self.loss_history[:, 1] = self.loss_history[:, 0]
        self.loss_history[:, 0] = losses.clone().detach()

        return weights


@META_ARCH_REGISTRY.register()
class DetectionAndSegmentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.dwa = DynamicWeightAveraging(num_tasks=2)
        self.register_buffer(
            "pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False
        )

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # Semantic Segmentation Head
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets,
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value,
                self.backbone.padding_constraints,
            ).tensor
        else:
            targets = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, targets)

        # Detection Pipeline (Proposal Generator + ROI Heads)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        detector_results, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        if self.training:
            losses = {}
            loss_seg = sum(
                v
                for v in sem_seg_losses.values()
                if torch.is_tensor(v) and v.numel() == 1
            )
            loss_det = sum(
                v
                for v in proposal_losses.values()
                if torch.is_tensor(v) and v.numel() == 1
            ) + sum(
                v
                for v in detector_losses.values()
                if torch.is_tensor(v) and v.numel() == 1
            )

            current_task_losses = torch.stack([loss_seg.detach(), loss_det.detach()])
            dwa_weights = self.dwa.update_weights(current_task_losses)
            weight_seg, weight_det = dwa_weights[0], dwa_weights[1]

            for k, v in sem_seg_losses.items():
                losses[k] = v * weight_seg
            for k, v in proposal_losses.items():
                losses[k] = v * weight_det
            for k, v in detector_losses.items():
                losses[k] = v * weight_det

            return losses

        processed_results = []
        for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            detector_r = detector_postprocess(detector_result, height, width)
            processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

        return processed_results


DETECTION_CLASSES = ["vehicle", "person", "traffic light", "traffic sign"]
SEGMENTATION_CLASSES = ["background", "drivable", "lane"]
OUTPUT_DIR = "/home/dylan/Documents/HydraNet/logs-merged"
LR = 3e-4
BATCH_SIZE = 4
STEPS = 300000  # 17500 steps per epoch
MODEL_CONFIG = "/home/dylan/Documents/HydraNet/detectron2/configs/Base-RCNN-FPN.yaml"
ZOO_WEIGHTS_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

for split in ["train", "val"]:
    dataset_name = f"BDD100k_{split}"
    DatasetCatalog.register(dataset_name, lambda split=split: get_dataset_dicts(split))
    MetadataCatalog.get(dataset_name).set(
        thing_classes=DETECTION_CLASSES,
        stuff_classes=SEGMENTATION_CLASSES,
        ignore_label=0,
    )

cfg = get_cfg()
cfg.merge_from_file(MODEL_CONFIG)
cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.DATASETS.TRAIN = ("BDD100k_train",)
cfg.DATASETS.TEST = ("BDD100k_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
cfg.SOLVER.BASE_LR = LR
cfg.SOLVER.MAX_ITER = STEPS
cfg.SOLVER.STEPS = []
cfg.SOLVER.CHECKPOINT_PERIOD = 10000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(DETECTION_CLASSES)
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(ZOO_WEIGHTS_FILE)

cfg.MODEL.META_ARCHITECTURE = "DetectionAndSegmentation"
cfg.MODEL.MASK_ON = False
cfg.VERSION = 2

cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"

cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.MODEL.RESNETS.DEPTH = 50

cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(SEGMENTATION_CLASSES)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

trainer.train()
