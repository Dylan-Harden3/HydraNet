import os
import torch
from PIL import Image
import numpy as np
import json
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms
import cv2
from detectron2.structures import BoxMode

# counts with these classes in train set:
# {'traffic light': 187871, 'traffic sign': 238270, 'vehicle': 751722, 'person': 96929}
CATEGORY_TO_CLASS = {
    # 0 for all vehicles
    "trailer": 0,
    "other vehicle": 0,
    "motorcycle": 0,
    "bus": 0,
    "train": 0,
    "truck": 0,
    "bicycle": 0,
    "car": 0,
    # 1 for all people
    "pedestrian": 1,
    "other person": 1,
    "rider": 1,
    # 2/3 for traffic light/sign
    "traffic light": 2,
    "traffic sign": 3
}


class BDD100KDataset(Dataset):
    def __init__(
        self, image_dir, drivable_dir, lane_dir, det_json, transform=None, resize=None, max_bboxes_per_image=50
    ):
        self.image_dir = image_dir
        self.drivable_dir = drivable_dir
        self.lane_dir = lane_dir
        self.transform = transform

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.drivable_filenames = sorted(os.listdir(self.drivable_dir))
        self.lane_filenames = sorted(os.listdir(self.lane_dir))

        self.resize = resize
        if resize:
            self.image_resize = v2.Compose(
                [
                    v2.Resize(resize, interpolation=v2.InterpolationMode.BILINEAR),
                ]
            )

            self.mask_resize = v2.Compose(
                [
                    v2.Resize(resize, interpolation=v2.InterpolationMode.NEAREST),
                ]
            )
        self.h_scale = resize[0] / 720 if resize else 1.0
        self.w_scale = resize[1] / 1280 if resize else 1.0

        with open(det_json, "r") as f:
            self.det_annotations = json.load(f)

        self.det_map = {ann["name"]: ann for ann in self.det_annotations}
        self.max_bboxes_per_image = max_bboxes_per_image

    def _extract_bboxes(self, det_annotation, max_bboxes_per_image):
        h_scale = self.resize[0] / 720
        w_scale = self.resize[1] / 1280
        bboxes = []
        for i, obj in enumerate(det_annotation.get("labels", [])):
            if i == max_bboxes_per_image:
                break
            bbox = obj["box2d"]
            bboxes.append([bbox["x1"] * w_scale, bbox["y1"] * h_scale, bbox["x2"] * w_scale, bbox["y2"] * h_scale, 1, CATEGORY_TO_CLASS[obj["category"]]])
        bboxes = (
            torch.tensor(bboxes, dtype=torch.float32)
            if bboxes
            else torch.zeros((0, 6), dtype=torch.float32)
        )
        num_objects = bboxes.shape[0]
        if num_objects < max_bboxes_per_image:
            padding = torch.zeros((max_bboxes_per_image - num_objects, 6), dtype=torch.float32)
            bboxes = torch.cat([bboxes, padding], dim=0)
        return bboxes

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Drivable
        drivable_path = os.path.join(self.drivable_dir, self.drivable_filenames[idx])
        drivable_mask = Image.open(drivable_path).convert("L")

        # Lane
        lane_path = os.path.join(self.lane_dir, self.lane_filenames[idx])
        lane_mask = Image.open(lane_path).convert("L")

        # Bounding Boxes
        bboxes = self._extract_bboxes(self.det_map.get(img_name, {}), self.max_bboxes_per_image)

        images = [image, drivable_mask, lane_mask]

        if self.image_resize:
            image = self.image_resize(image)
            drivable_mask = self.mask_resize(drivable_mask)
            lane_mask = self.mask_resize(lane_mask)

        drivable_mask_np = np.array(drivable_mask)
        lane_mask_np = np.array(lane_mask)
        
        # Thresholding to create binary masks
        _, drivable_bin = cv2.threshold(drivable_mask_np, 1, 1, cv2.THRESH_BINARY)
        _, lane_bin = cv2.threshold(lane_mask_np, 1, 2, cv2.THRESH_BINARY)
        
        combined_mask = drivable_bin.copy()
        combined_mask[lane_bin == 2] = 2
        
        tensor_transform = torchvision.transforms.ToTensor()
        image = tensor_transform(image)
        segmentation_mask = torch.from_numpy(combined_mask).long()
        
        normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        return image, segmentation_mask, bboxes

def get_dataset_dicts(split, category_to_class=CATEGORY_TO_CLASS):
    if split == "train":
        image_dir="100k_images_train/bdd100k/images/100k/train/"
        drivable_dir="da_seg_annotations/bdd_seg_gt/train/"
        lane_dir="ll_seg_annotations/bdd_lane_gt/train"
        det_json="bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_train.json"
    elif split == "val":
        image_dir="100k_images_val/bdd100k/images/100k/val/"
        drivable_dir="da_seg_annotations/bdd_seg_gt/train/"
        lane_dir="ll_seg_annotations/bdd_lane_gt/train"
        det_json="bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_val.json"
    elif split == "test":
        pass

    image_filenames = sorted(os.listdir(image_dir))
    drivable_filenames = sorted(os.listdir(drivable_dir))
    lane_filenames = sorted(os.listdir(lane_dir))

    with open(det_json, "r") as f:
        det_annotations = json.load(f)
        det_map = {ann["name"]: ann for ann in det_annotations}

    dataset_dicts = []
    for idx, filename in enumerate(image_filenames):
        record = {}
        file_path = os.path.join(image_dir, filename)

        record["file_name"] = file_path
        record["image_id"] = idx
        record["height"] = 720
        record["width"] = 1280

        annotations = []
        obj = det_map.get(filename, {})
        for label in obj.get("labels", []):
            anno = {}
            bbox = label["box2d"]
            anno["bbox"] = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            anno["category_id"] = CATEGORY_TO_CLASS[label["category"]]
            annotations.append(anno)
        record["annotations"] = annotations
        dataset_dicts.append(record)
    return dataset_dicts