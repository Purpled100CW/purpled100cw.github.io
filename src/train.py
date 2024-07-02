import os
import torch
from torch.utils.data import DataLoader
from dataset import RoadDataset, transform
from model import RoadNet, train_model
import torch.optim as optim
import torch.nn as nn

data_dir = 'data'
batch_size = 8
num_epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_datasets = {
    'train': RoadDataset(os.path.join(data_dir, 'train/images'), os.path.join(data_dir, 'train/labels'), transform),
    'val': RoadDataset(os.path.join(data_dir, 'val/images'), os.path.join(data_dir, 'val/labels'), transform)
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)
}

model = RoadNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model = train_model(model, dataloaders, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), 'models/road_detection_model.pth')
