import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, 1)
        self.net = base
    def forward(self,x): return self.net(x).squeeze(1)

def predict(image_path, ckpt="best_coin_resnet.pth"):
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0)
    model = ResNetRegressor()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred = model(x)
    return round(pred.item())

print(predict("dataset/coins_images/coins_images/all_coins/fa5950f31e.jpg"))
