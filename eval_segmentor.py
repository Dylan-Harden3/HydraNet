from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from dataset import get_dataset_dicts
from detectron2.evaluation import inference_on_dataset, SemSegEvaluator
from detectron2.data import build_detection_test_loader

SEGMENTATION_CLASSES = ["background", "drivable", "lane"]
MODEL_CONFIG = "/home/dylan/Documents/HydraNet/detectron2/configs/Base-RCNN-FPN.yaml"
OUTPUT_DIR = "/home/dylan/Documents/HydraNet/sem_seg_logs"
BATCH_SIZE = 4

for split in ["train", "val"]:
    dataset_name = f"BDD100k_{split}"
    DatasetCatalog.register(dataset_name, lambda split=split: get_dataset_dicts(split))
    MetadataCatalog.get(dataset_name).set(
        thing_classes=SEGMENTATION_CLASSES,
        stuff_classes=SEGMENTATION_CLASSES,
        ignore_label=0,
    )

WEIGHTS_FILE = "/home/dylan/Documents/HydraNet/sem_seg_logs/"
OUTPUT_RESULTS_DIR = "./inference_results"
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)


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
cfg.MODEL.WEIGHTS = WEIGHTS_FILE
cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
cfg.MODEL.DEVICE = "cuda"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

evaluator = SemSegEvaluator("BDD100k_val", False, output_dir="./output_seg/")
val_loader = build_detection_test_loader(cfg, "BDD100k_val")

print(inference_on_dataset(trainer.model, val_loader, evaluator))
