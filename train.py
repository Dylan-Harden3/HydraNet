import torch.nn as nn
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from hydranet import HydraNet
from tqdm import tqdm

class UTKDataset(Dataset):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.root_path = "archive/UTKFace/"
        image_paths = os.listdir(self.root_path)
        self.image_paths = image_paths
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []
        for path in image_paths:
            filename = path[:8].split("_")
            if len(filename) == 4:
                self.images.append(path)
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.root_path + self.images[index]).convert("RGB")
        img = self.transform(img)

        age = self.ages[index]
        gender = self.genders[index]
        race = self.races[index]

        sample = {"image": img, "age": age, "gender": gender, "race": race}
        return sample


epochs = 50
batch_size = 128
lr = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HydraNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
age_criteroen = nn.L1Loss()
race_criteroen = nn.CrossEntropyLoss()
gender_criteroen = nn.BCEWithLogitsLoss()

full_dataset = UTKDataset()
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_age_loss = 0
    total_race_loss = 0
    total_gender_loss = 0

    for batch in tqdm(train_dataloader):
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
        f"Epoch {epoch} Loss: {total_loss/len(train_dataloader)} Age Loss: {total_age_loss/len(train_dataloader)} Race Loss: {total_race_loss/len(train_dataloader)} Gender Loss: {total_gender_loss/len(train_dataloader)}"
    )
