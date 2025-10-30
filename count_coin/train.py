import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image

# Dataset class
class CoinCountDataset(Dataset):
    def __init__(self, root_dir, csv_file, split="train", val_size=0.15, test_size=0.15, random_state=42):
        self.root = Path(root_dir)
        df = pd.read_csv(csv_file)
        df["path"] = df.apply(lambda r: str(self.root / r["folder"] / r["image_name"]), axis=1)

        # train / val / test split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df, val_df  = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state)
        if split == "train": self.df = train_df
        elif split == "val": self.df = val_df
        else: self.df = test_df

        # transforms
        base_tfms = [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
        if split == "train":
            self.tfms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.2,0.1),
                *base_tfms
            ])
        else:
            self.tfms = transforms.Compose(base_tfms)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.tfms(img)
        count = torch.tensor([row["coins_count"]], dtype=torch.float32)
        return img, count

# Model
class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, 1)
        self.net = base
    def forward(self, x): return self.net(x).squeeze(1)

# Training
def train_model(root_dir, csv_file, epochs=15, batch_size=16, lr=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = CoinCountDataset(root_dir, csv_file, "train")
    ds_val   = CoinCountDataset(root_dir, csv_file, "val")
    ds_test  = CoinCountDataset(root_dir, csv_file, "test")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test  = DataLoader(ds_test, batch_size=1, shuffle=False)

    model = ResNetRegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_mae = float("inf")

    for epoch in range(1, epochs+1):
        # ---- train ----
        model.train()
        for imgs, counts in dl_train:
            imgs, counts = imgs.to(device), counts.to(device)
            pred = model(imgs)
            loss = F.l1_loss(pred, counts.squeeze())  # MAE
            opt.zero_grad(); loss.backward(); opt.step()
        # ---- val ----
        model.eval()
        mae = 0
        with torch.no_grad():
            for imgs, counts in dl_val:
                imgs, counts = imgs.to(device), counts.to(device)
                pred = model(imgs)
                mae += (pred - counts.squeeze()).abs().sum().item()
        mae /= len(ds_val)
        print(f"Epoch {epoch:02d} | val MAE={mae:.3f}")
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "best_coin_resnet.pth")
            print("✓ Saved new best model")
    print("Training done. Best val MAE:", best_mae)

    # ---- evaluate on test ----
    model.load_state_dict(torch.load("best_coin_resnet.pth"))
    model.eval()
    total_err = 0
    with torch.no_grad():
        for imgs, counts in dl_test:
            imgs, counts = imgs.to(device), counts.to(device)
            pred = model(imgs)
            total_err += (pred - counts.squeeze()).abs().item()
    print(f"Test MAE = {total_err / len(ds_test):.3f}")

if __name__ == "__main__":
    train_model(
        root_dir="dataset/coins_images/coins_images",
        csv_file="dataset/coins_count_values.csv",
        epochs=20,
        batch_size=8,
        lr=1e-4
    )
