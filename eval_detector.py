from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from dataset import get_dataset_dicts

DETECTION_CLASSES = ["vehicle", "person", "traffic light", "traffic sign"]
MODEL_CONFIG = "/home/dylan/Documents/HydraNet/detectron2/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
WEIGHTS_FILE = "/home/dylan/Documents/HydraNet/logs/model_0099999.pth"
splits = ["train", "val"]

for split in splits:
    dataset_name = f"BDD100k_{split}"
    DatasetCatalog.register(dataset_name, lambda split=split: get_dataset_dicts(split))
    MetadataCatalog.get(dataset_name).set(
        thing_classes=DETECTION_CLASSES,
    )

cfg = get_cfg()
cfg.merge_from_file(MODEL_CONFIG)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.DATASETS.TRAIN = ("BDD100k_train",)
cfg.DATASETS.TEST = ("BDD100k_val",)
cfg.MODEL.WEIGHTS = WEIGHTS_FILE
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(DETECTION_CLASSES)
cfg.MODEL.DEVICE = "cuda"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
evaluator = COCOEvaluator("BDD100k_val", ("bbox",), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "BDD100k_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
