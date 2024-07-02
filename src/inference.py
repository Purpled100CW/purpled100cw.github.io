import cv2
import torch
import numpy as np
from model import load_model

model_path = 'models/road_detection_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(image_path, model):
    image = cv2.imread(image_path)
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        outputs = outputs.cpu().numpy().squeeze()

    outputs = cv2.resize(outputs, (orig.shape[1], orig.shape[0]))
    outputs = (outputs > 0.5).astype(np.uint8) * 255
    overlay = cv2.addWeighted(orig, 0.5, cv2.cvtColor(outputs, cv2.COLOR_GRAY2BGR), 0.5, 0)
    
    return overlay

model = load_model(model_path).to(device)
output_image = inference('path_to_image.jpg', model)
cv2.imwrite('output.jpg', output_image)
