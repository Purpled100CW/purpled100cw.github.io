import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class RoadDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        label_name = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.png'))
        
        image = cv2.imread(img_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
])
