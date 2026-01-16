import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/surface_defect_model.pt"

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, 4
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

IDX_TO_CLASS = {0:"good",1:"scratch",2:"oil",3:"stain"}

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@torch.no_grad()
def predict_surface_patch(patch):
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    img = tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(img), dim=1)[0].cpu().numpy()
    i = int(np.argmax(probs))
    return IDX_TO_CLASS[i], float(probs[i])

def surface_score_from_patch_preds(preds):
    defects = [p for p in preds if p["cls"]!="good" and p["conf"]>0.7]
    r = len(defects)/len(preds)

    if r <= 0.05: return 10
    if r <= 0.10: return 9
    if r <= 0.20: return 8
    if r <= 0.30: return 7
    if r <= 0.40: return 6
    if r <= 0.55: return 5
    if r <= 0.70: return 4
    if r <= 0.85: return 3
    return 2
