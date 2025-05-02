from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from dataset import get_dataset_dicts
from detectron2.evaluation import inference_on_dataset, SemSegEvaluator
from detectron2.data import build_detection_test_loader
import json
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
from detectron2.evaluation import COCOEvaluator


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
            losses.update(sem_seg_losses)
            losses.update(proposal_losses)
            losses.update(detector_losses)
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


SEGMENTATION_CLASSES = ["background", "drivable", "lane"]
DETECTION_CLASSES = ["vehicle", "person", "traffic light", "traffic sign"]
OUTPUT_RESULTS_DIR = "./merged-results/"
MODEL_CONFIG = "/home/dylan/Documents/HydraNet/detectron2/configs/Base-RCNN-FPN.yaml"
OUTPUT_DIR = "/home/dylan/Documents/HydraNet/logs-merged-out"
BATCH_SIZE = 4
CHECKPOINT_DIR = "/home/dylan/Documents/HydraNet/logs-merged/"
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

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
cfg.DATASETS.TRAIN = ("BDD100k_train",)
cfg.DATASETS.TEST = ("BDD100k_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
cfg.SOLVER.STEPS = []
cfg.SOLVER.CHECKPOINT_PERIOD = 10000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(DETECTION_CLASSES)
cfg.MODEL.DEVICE = "cuda"

cfg.MODEL.META_ARCHITECTURE = "DetectionAndSegmentation"
cfg.MODEL.MASK_ON = False
cfg.VERSION = 2

cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"

cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.MODEL.RESNETS.DEPTH = 50

cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(SEGMENTATION_CLASSES)

cfg.MODEL.WEIGHTS = CHECKPOINT_DIR + "model_0169999.pth"

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

evaluator = SemSegEvaluator("BDD100k_val", False, output_dir=OUTPUT_RESULTS_DIR)
val_loader = build_detection_test_loader(cfg, "BDD100k_val")
seg_results = inference_on_dataset(trainer.model, val_loader, evaluator)
print("Seg results:", seg_results)

evaluator = COCOEvaluator("BDD100k_val", ("bbox",), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "BDD100k_val")
det_results = inference_on_dataset(trainer.model, val_loader, evaluator)
print("Detection results:", det_results)