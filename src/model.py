import torch
import torch.nn as nn

import torchvision.models as models

global device
device = ("cuda" if torch.cuda.is_available() else "cpu")

class RoadNet(nn.Module):
    def __init__(self, num_classes=1):
        super(RoadNet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base_model(x)
        return x

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

    return model

def load_model(model_path):
    model = RoadNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
