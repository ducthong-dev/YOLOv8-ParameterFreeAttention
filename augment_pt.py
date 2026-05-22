import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import LeafDataset, test_aug
from PIL import Image


def save_augmented_test_images(test_dataset, save_dir="augmented_test_images"):
    """Save all augmented images from the test set to a directory for evaluation, using class names as folder names."""
    os.makedirs(save_dir, exist_ok=True)
    # Get class2idx mapping and invert it
    class2idx = getattr(test_dataset, "class2idx", None)
    if class2idx is not None:
        idx2class = {v: k for k, v in class2idx.items()}
    else:
        idx2class = None
    for idx in range(len(test_dataset)):
        image, label = test_dataset[idx]
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array(
            [0.485, 0.456, 0.406]
        )
        img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        # Use class name if available
        class_name = idx2class[label] if idx2class is not None else str(label)
        label_dir = os.path.join(save_dir, class_name)
        os.makedirs(label_dir, exist_ok=True)
        img_pil.save(os.path.join(label_dir, f"img_{idx}.png"))
    print(
        f"Saved {len(test_dataset)} augmented images to {save_dir}/[class_name]/img_*.png"
    )


def main():
    test_dir = "dataset/Plant_leaf_diseases_dataset/test"
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist!")
        return
    test_dataset = LeafDataset(test_dir, transform=test_aug)
    print(f"Test samples: {len(test_dataset)}")
    save_augmented_test_images(
        test_dataset,
        save_dir="dataset/Plant_leaf_diseases_dataset/augmented_test_images_hardest",
    )


if __name__ == "__main__":
    main()
