from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from dataset import get_dataset_dicts

DETECTION_CLASSES = ["vehicle", "person", "traffic light", "traffic sign"]
MODEL_CONFIG = "/home/dylan/Documents/HydraNet/detectron2/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
ZOO_WEIGHTS_FILE= "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
OUTPUT_DIR = "/home/dylan/Documents/HydraNet/logs"
LR = 3e-4
BATCH_SIZE = 4
STEPS = 300000 # 17500 steps per epoch

for split in ["train", "val"]:
    dataset_name = f"BDD100k_{split}"
    DatasetCatalog.register(
        dataset_name,
        lambda split=split: get_dataset_dicts(split)
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=DETECTION_CLASSES,
    )
'''
TODO:
- Freeze Backbone/FPN and train segmentation head
- Train both with DWA/mess with loss
- Script for getting the best validation loss model/plotting loss
- Visualize on BDD100k video clips
'''

cfg = get_cfg()
cfg.merge_from_file(MODEL_CONFIG)
cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.DATASETS.TRAIN = ("BDD100k_train",)
cfg.DATASETS.TEST = ("BDD100k_val", )
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(ZOO_WEIGHTS_FILE)
cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
cfg.SOLVER.BASE_LR = LR
cfg.SOLVER.MAX_ITER = STEPS
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(DETECTION_CLASSES)
cfg.MODEL.DEVICE = 'cuda'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()