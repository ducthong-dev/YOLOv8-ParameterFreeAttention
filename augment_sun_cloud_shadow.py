import cv2
import os
from test_augmentation import add_sun_cloud_shadow
import concurrent.futures
from tqdm import tqdm

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
        aug_img = add_sun_cloud_shadow(img)
        cv2.imwrite(out_path, aug_img)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_one, image_tasks), total=len(image_tasks), desc="Sun/Cloud Shadow"))

if __name__ == "__main__":
    input_dir = "dataset/Plant_leaf_diseases_dataset/test"
    output_dir = "dataset/Plant_leaf_diseases_dataset/sun_cloud_shadow"
    process_folder(input_dir, output_dir)
