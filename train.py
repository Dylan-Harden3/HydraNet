import torch.nn as nn
import os
import torch
from PIL import Image
from hydranet import HydraNet
from hydranetfactory import HydraNetFactory, HydraNetConfig
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, filepath):
  checkpoint = {
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'train_losses': train_losses,
          'val_losses': val_losses,
      }
  torch.save(checkpoint, f'{filepath}/model_checkpoint_{epoch}.pt')
  torch.save(model.state_dict(), f'{filepath}/model_checkpoint_{epoch}.pt')

def load_checkpoint(filepath, model, optimizer, scheduler):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    return model, optimizer, scheduler, start_epoch, train_losses, val_losses

class BDD100KDataset(Dataset):
    def __init__(self, image_dir, drivable_dir, lane_dir, det_json, transform=None):
        self.image_dir = image_dir
        self.drivable_dir = drivable_dir
        self.lane_dir = lane_dir
        self.transform = transform

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.drivable_filenames = sorted(os.listdir(self.drivable_dir))
        self.lane_filenames = sorted(os.listdir(self.lane_dir))

        with open(det_json, 'r') as f:
            self.det_annotations = json.load(f)

        self.det_map = {ann["name"]: ann for ann in self.det_annotations}

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
        drivable_mask = Image.open(drivable_path).convert("RGB")

        # Lane Path
        lane_path = os.path.join(self.lane_dir, self.lane_filenames[idx])
        lane_mask = Image.open(lane_path)

        # # Object Detection
        # det_annotation = self.det_map.get(img_name, {})
        # bboxes = self._extract_bboxes(det_annotation)

        if self.transform:
            image = self.transform(image)
            drivable_mask = self.transform(drivable_mask)
            lane_mask = self.transform(lane_mask)

        image = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)

        return image, drivable_mask, lane_mask

transforms = v2.Compose([v2.Resize((256, 256)),
                         v2.ToImage(),
                         v2.ToDtype(torch.float32, scale=True)
                        ])

train_dataset = BDD100KDataset(
    image_dir="100k_images_train/bdd100k/images/100k/train/",
    drivable_dir="bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/colormaps/train/",
    lane_dir="bdd100k_lane_labels_trainval/bdd100k/labels/lane/masks/train/",
    det_json="bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_train.json",
    transform=transforms,
)

val_dataset = BDD100KDataset(
    image_dir="100k_images_val/bdd100k/images/100k/val/",
    drivable_dir="bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/colormaps/val/",
    lane_dir="bdd100k_lane_labels_trainval/bdd100k/labels/lane/masks/val/",
    det_json="bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_val.json",
    transform=transforms,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
        self.loss_history[:, 0] = torch.tensor(losses)

        return weights

epochs = 30
batch_size = 32
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

config = HydraNetConfig(
    regnet="regnet_y_400mf",
    image_size=(3, 256, 256),
    bifpn_channels=45,
    n_bifpn_blocks=3,
)

model = HydraNetFactory(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

lane_criterion = nn.BCEWithLogitsLoss()
drivable_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.SmoothL1Loss()

dwa = DynamicWeightAveraging(num_tasks=2)

train_losses = []
train_lane_losses = []
train_drivable_losses = []
val_losses = []
val_lane_losses = []
val_drivable_losses = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_train_lane_loss = 0
    total_train_drivable_loss = 0
    total_train_bbox_loss = 0

    for batch, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        images, drivable_masks, lane_masks = data
        images = images.to(device)
        drivable_masks = drivable_masks.to(device)
        lane_masks = lane_masks.to(device)

        p1, p2, p3, p4, lane_det_output, drivable_area_output = model(images)

        drivable_loss = drivable_criterion(drivable_area_output, drivable_masks)
        bbox_loss = 0
        # bbox_loss = bbox_criterion(torch.cat([p1, p2, p3, p4], dim=1), bboxes)
        lane_loss = lane_criterion(lane_det_output, lane_masks)

        weights = dwa.update_weights([lane_loss.item(), drivable_loss.item()])
        total_loss = weights[0] * lane_loss + weights[1] * drivable_loss

        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss.item()
        total_train_drivable_loss += drivable_loss.item()
        #total_train_bbox_loss += bbox_loss.item()
        total_train_lane_loss += lane_loss.item()

        if (batch + 1) % 100 == 0:
            loss = total_train_loss / (batch + 1)
            lane_loss_current = total_train_lane_loss / (batch + 1)
            drivable_loss_current = total_train_drivable_loss / (batch + 1)
            bbox_loss_current = total_train_bbox_loss / (batch + 1)
            current = (batch + 1) * batch_size

            print(f"Training loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")
            print(f"Lane Loss: {lane_loss_current:>7f} | Drivable Loss: {drivable_loss_current:>7f} | BBox Loss: {bbox_loss_current:>7f}")

    print(
        f"Epoch {epoch+1} | Total Train Loss: {total_train_loss/len(train_loader):.4f} | "
        f"Train Lane Loss: {total_train_lane_loss/len(train_loader):.4f} | "
        f"Train Drivable Loss: {total_train_drivable_loss/len(train_loader):.4f} | "
        f"Train BBox Loss: {total_train_bbox_loss/len(train_loader):.4f}"
    )

    model.eval()
    total_val_loss = 0
    total_val_lane_loss = 0
    total_val_drivable_loss = 0
    total_val_bbox_loss = 0

    with torch.no_grad():
        for images, drivable_masks, lane_masks in tqdm(val_loader):
            images = images.to(device)
            drivable_masks = drivable_masks.to(device)
            lane_masks = lane_masks.to(device)

            p1, p2, p3, p4, lane_det_output, drivable_area_output = model(images)

            drivable_loss = drivable_criterion(drivable_area_output, drivable_masks)
            bbox_loss = 0
            # bbox_loss = bbox_criterion(torch.cat([p1, p2, p3, p4], dim=1), bboxes)
            lane_loss = lane_criterion(lane_det_output, lane_masks)

            total_loss = drivable_loss + bbox_loss + lane_loss

            total_val_loss += total_loss.item()
            total_val_drivable_loss += drivable_loss.item()
            #total_val_bbox_loss += bbox_loss.item()
            total_val_lane_loss += lane_loss.item()

    print(
        f"Total Val Loss: {total_val_loss/len(val_loader):.4f} | "
        f"Val Lane Loss: {total_val_lane_loss/len(val_loader):.4f} | "
        f"Val Drivable Loss: {total_val_drivable_loss/len(val_loader):.4f} | "
        f"Val BBox Loss: {total_val_bbox_loss/len(val_loader):.4f}"
    )

    train_losses.append(total_train_loss/len(train_loader))
    train_lane_losses.append(total_train_lane_loss/len(train_loader))
    train_drivable_losses.append(total_train_drivable_loss/len(train_loader))
    val_losses.append(total_val_loss/len(val_loader))
    val_lane_losses.append(total_val_lane_loss/len(train_loader))
    val_drivable_losses.append(total_val_drivable_loss/len(train_loader))

    scheduler.step()
    save_checkpoint(model, optimizer, scheduler, epoch+1, train_losses, val_losses)