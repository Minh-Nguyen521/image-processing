import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import cv2 as cv
import numpy as np

def show_tensor(tensor, win="image"):
    # if batched [1,3,H,W] -> [3,H,W]
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    # undo normalization
    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std  = torch.tensor([0.229,0.224,0.225])[:,None,None]
    img = tensor * std + mean
    # clip to [0,1]
    img = torch.clamp(img, 0, 1)
    # convert to numpy, swap to HWC, scale 0–255
    img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
    # BGR for OpenCV display
    img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow(win, img_bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()

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

print(predict("./2c8f0e58fb.jpg"))
