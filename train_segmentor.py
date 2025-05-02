from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from dataset import get_dataset_dicts

SEGMENTATION_CLASSES = ["background", "drivable", "lane"]
MODEL_CONFIG = "/home/dylan/Documents/HydraNet/detectron2/configs/Base-RCNN-FPN.yaml"
OUTPUT_DIR = "/home/dylan/Documents/HydraNet/sem_seg_logs"
LR = 3e-4
BATCH_SIZE = 4
STEPS = 300000  # 17500 steps per epoch

for split in ["train", "val"]:
    dataset_name = f"BDD100k_{split}"
    DatasetCatalog.register(dataset_name, lambda split=split: get_dataset_dicts(split))
    MetadataCatalog.get(dataset_name).set(
        thing_classes=SEGMENTATION_CLASSES,
    )

cfg = get_cfg()
cfg.merge_from_file(MODEL_CONFIG)
cfg.MODEL.META_ARCHITECTURE = "SemanticSegmentor"
cfg.MODEL.MASK_ON = False
cfg.VERSION = 2

cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"

cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.MODEL.RESNETS.DEPTH = 50

cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(SEGMENTATION_CLASSES)
cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.DATASETS.TRAIN = ("BDD100k_train",)
cfg.DATASETS.TEST = ("BDD100k_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "/home/dylan/Documents/HydraNet/logs/model_0099999.pth"
cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
cfg.SOLVER.MAX_ITER = STEPS
cfg.SOLVER.STEPS = []
cfg.SOLVER.BASE_LR = LR
cfg.MODEL.DEVICE = "cuda"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
for param in trainer.model.backbone.parameters():
    param.requires_grad = False

trainer.train()
