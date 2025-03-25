import torch.nn as nn
import os
import torch
from PIL import Image
from hydranet import HydraNet
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


class BDD100KDataset(Dataset):
    def __init__(self, image_dir, drivable_dir, det_json, pose_json, transform=None):
        self.image_dir = image_dir
        self.drivable_dir = drivable_dir
        self.transform = transform

        self.image_filenames = sorted(os.listdir(image_dir))
        self.drivable_filenames = sorted(os.listdir(drivable_dir))

        with open(det_json, 'r') as f:
            self.det_annotations = json.load(f)
        
        with open(pose_json, 'r') as f:
            pose_data = json.load(f)
            self.pose_annotations = pose_data.get("frames", [])

        self.det_map = {ann["name"]: ann for ann in self.det_annotations}
        self.pose_map = {ann["name"]: ann for ann in self.pose_annotations}

    def _extract_keypoints(self, pose_annotation):
        if not pose_annotation or "labels" not in pose_annotation or pose_annotation["labels"] is None:
            return torch.zeros((0, 18, 2), dtype=torch.float32)

        keypoints = []
        for person in pose_annotation["labels"]:
            if "graph" in person and "nodes" in person["graph"]:
                person_keypoints = [node["location"] for node in person["graph"]["nodes"]]
                keypoints.append(person_keypoints)

        return torch.tensor(keypoints, dtype=torch.float32) if keypoints else torch.zeros((0, 18, 2), dtype=torch.float32)

    def _extract_bboxes(self, det_annotation):
        bboxes = []
        for obj in det_annotation.get("labels", []):
            bbox = obj["box2d"]
            bboxes.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32)
        return bboxes

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Drivable Path
        drivable_path = os.path.join(self.drivable_dir, self.drivable_filenames[idx])
        drivable_mask = Image.open(drivable_path)

        # Object Detection
        det_annotation = self.det_map.get(img_name, {})
        bboxes = self._extract_bboxes(det_annotation)
        
        # Pose Estimation
        pose_annotation = self.pose_map.get(img_name, {})
        if not isinstance(pose_annotation, dict):
            pose_annotation = {}
        keypoints = self._extract_keypoints(pose_annotation)

        if self.transform:
            image = self.transform(image)
            drivable_mask = self.transform(drivable_mask)

        image = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)

        return image, drivable_mask, bboxes, keypoints

def collate_fn(batch):
    images, drivable_masks, bboxes, keypoints = zip(*batch)

    images = torch.stack(images, dim=0)
    drivable_masks = torch.stack(drivable_masks, dim=0)

    return images, drivable_masks, bboxes, keypoints

transforms = v2.Compose([v2.Resize((360, 540)), 
                         v2.ToImage(),
                         v2.ToDtype(torch.float32, scale=True)
                        ])

train_dataset = BDD100KDataset(
    image_dir="/mnt/c/Users/User/Documents/Homework/CSCE 753 (CVRP)/HydraNet/100k_images_train/bdd100k/images/100k/train/",
    drivable_dir="/mnt/c/Users/User/Documents/Homework/CSCE 753 (CVRP)/HydraNet/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/masks/train/",
    det_json="/mnt/c/Users/User/Documents/Homework/CSCE 753 (CVRP)/HydraNet/bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_train.json",
    pose_json="/mnt/c/Users/User/Documents/Homework/CSCE 753 (CVRP)/HydraNet/bdd100k_pose_labels_trainval/bdd100k/labels/pose_21/pose_train.json",
    transform=transforms
)

val_dataset = BDD100KDataset(
    image_dir="/mnt/c/Users/User/Documents/Homework/CSCE 753 (CVRP)/HydraNet/100k_images_val/bdd100k/images/100k/val/",
    drivable_dir="/mnt/c/Users/User/Documents/Homework/CSCE 753 (CVRP)/HydraNet/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/masks/val/",
    det_json="/mnt/c/Users/User/Documents/Homework/CSCE 753 (CVRP)/HydraNet/bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_val.json",
    pose_json="/mnt/c/Users/User/Documents/Homework/CSCE 753 (CVRP)/HydraNet/bdd100k_pose_labels_trainval/bdd100k/labels/pose_21/pose_val.json",
    transform=transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

epochs = 50
batch_size = 128
lr = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HydraNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
age_criteroen = nn.L1Loss()
race_criteroen = nn.CrossEntropyLoss()
gender_criteroen = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_age_loss = 0
    total_race_loss = 0
    total_gender_loss = 0

    for batch in tqdm(train_loader):
        images = batch["image"].to(device)
        ages = batch["age"].to(device)
        races = batch["race"].to(device)
        genders = batch["gender"].to(device)

        preds = model(images)
        age_loss = age_criteroen(preds["age"], ages.unsqueeze(1).float())
        race_loss = race_criteroen(preds["race"], races)
        gender_loss = gender_criteroen(preds["gender"], genders.unsqueeze(1).float())

        loss = age_loss + race_loss + gender_loss

        total_loss += loss.item()
        total_age_loss += age_loss.item()
        total_race_loss += race_loss.item()
        total_gender_loss += gender_loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(
        f"Epoch {epoch} Loss: {total_loss/len(train_loader)} Age Loss: {total_age_loss/len(train_loader)} Race Loss: {total_race_loss/len(train_loader)} Gender Loss: {total_gender_loss/len(train_loader)}"
    )
