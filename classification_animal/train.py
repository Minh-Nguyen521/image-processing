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
class AnimalDataset(Dataset):
    def __init__(self, root_dir, csv_file, split="train", val_size=0.15, test_size=0.15, random_state=42):
        self.root = Path(root_dir)
        df = pd.read_csv(csv_file)
        df["path"] = df.apply(lambda r: str(self.root / r["folder"] / r["image_name"]), axis=1)

        # train / val / test split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["animal_label"])
        train_df, val_df  = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_df["animal_label"])
        
        if split == "train": 
            self.df = train_df
        elif split == "val": 
            self.df = val_df
        else: 
            self.df = test_df

        base_tfms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        label = torch.tensor(row["animal_label"], dtype=torch.long)
        return img, label

# Model
class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, 11)
        self.net = base
    def forward(self, x): 
        return self.net(x)

# Training
def train_model(root_dir, csv_file, epochs=15, batch_size=16, lr=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = AnimalDataset(root_dir, csv_file, "train")
    ds_val   = AnimalDataset(root_dir, csv_file, "val")
    ds_test  = AnimalDataset(root_dir, csv_file, "test")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test  = DataLoader(ds_test, batch_size=1, shuffle=False)

    model = ResNetClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in dl_train:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device).long()             # [B], ints

            logits = model(imgs)                          # [B, K]
            print("logits", logits.shape, "labels", labels.shape, labels.dtype) # Expect logits [B, C], labels [B], long
            loss = criterion(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        n = 0
        with torch.inference_mode():
            for imgs, labels in dl_val:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device).long()
                logits = model(imgs)                      # [B, K]
                preds = logits.argmax(dim=1)              # [B]
                correct += (preds == labels).sum().item()
                n += labels.size(0)

        acc = correct / n
        print(f"Epoch {epoch:02d} | val Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_animal_cls_resnet.pth")
            print("✓ Saved new best model")

    print("Training done. Best val Acc:", best_acc)

    # ---- evaluate on test ----
    model.load_state_dict(torch.load("best_animal_cls_resnet.pth"))
    model.eval()
    total_err = 0
    with torch.no_grad():
        for imgs, counts in dl_test:
            imgs, counts = imgs.to(device), counts.to(device)
            pred = model(imgs)
            total_err += (pred - counts.squeeze()).abs().item()
    print(f"Test MAE = {total_err / len(ds_test):.3f}")
    print(f"Accuracy within ±0.5: {sum(1 for imgs, counts in dl_test if abs(model(imgs.to(device)) - counts.to(device).squeeze()) <= 0.5) / len(ds_test):.3%}")

if __name__ == "__main__":
    train_model(
        root_dir="dataset/raw-img",
        csv_file="animals_dataset.csv",
        epochs=10,
        batch_size=8,
        lr=1e-4
    )
