import os
import cv2
import glob
import torch
from torch.utils.data import Dataset

class TeaLeafDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(data_dir, split, "images")
        self.labels_dir = os.path.join(data_dir, split, "labels")

        self.image_paths = sorted(
            glob.glob(os.path.join(self.images_dir, "*.jpg"))
        )

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load Image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV loads as BGR, convert to RGB

        # Load Label from corresponding text file
        file_name = os.path.basename(img_path)
        label_name = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        target_class = 0 # Default to 0 if file is empty
        
        try:
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        target_class = int(line.split()[0])
        except Exception as e:
            print(f"Warning: Could not read label for {img_path}: {e}")

        # Apply Augmentations (Albumentations)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, target_class