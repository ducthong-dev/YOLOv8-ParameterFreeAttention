import cv2
import os
import numpy as np
import concurrent.futures
from tqdm import tqdm

def background_blur(image, ksize=21):
    # Simple background blur using color threshold for green leaves
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    fg = cv2.bitwise_and(image, image, mask=mask)
    bg = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
    return cv2.add(fg, bg)

def process_folder(input_dir, output_dir, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    image_tasks = []
    for root, _, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel)
        os.makedirs(out_root, exist_ok=True)
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, f)
                out_path = os.path.join(out_root, f)
                image_tasks.append((img_path, out_path))

    def process_one(args):
        img_path, out_path = args
        img = cv2.imread(img_path)
        if img is None:
            return False
        aug_img = background_blur(img)
        cv2.imwrite(out_path, aug_img)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_one, image_tasks), total=len(image_tasks), desc="Background Blur"))

if __name__ == "__main__":
    input_dir = "dataset/Plant_leaf_diseases_dataset/test"
    output_dir = "dataset/background_blur"
    process_folder(input_dir, output_dir)
