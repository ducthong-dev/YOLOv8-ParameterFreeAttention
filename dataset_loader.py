import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Augmentation pipelines
train_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Blur(blur_limit=3, p=0.2),
        A.GaussNoise(variance_limit=(10, 50), p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_aug = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(),
        ToTensorV2(),
    ]
)


test_aug = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.15, p=0.4),
        A.GaussNoise(var_limit=(10, 60), p=0.4),
        A.Blur(blur_limit=5, p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.MedianBlur(blur_limit=5, p=0.2),
        A.CoarseDropout(
            max_holes=4,
            max_height=48,
            max_width=48,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=0,
            p=0.4,
        ),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        A.Solarize(threshold=128, p=0.1),
        A.Posterize(num_bits=4, p=0.1),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.15),
        A.RandomRain(
            blur_value=2,
            brightness_coefficient=0.9,
            drop_width=1,
            drop_length=20,
            p=0.15,
        ),
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.15,
        ),
        A.Normalize(),
        ToTensorV2(),
    ]
)


# Custom Dataset
class LeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class2idx = {}
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class2idx[class_name] = len(self.class2idx)
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(
                            (
                                os.path.join(class_path, img_file),
                                self.class2idx[class_name],
                            )
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label
