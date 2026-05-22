import os

# Paths to label.names and output data.yaml
label_names_file = "yolo_labels_original_plant_leaf_disease_vit_0.1-0.9/label.names"
output_yaml_file = "yolo_labels_original_plant_leaf_disease_vit_0.1-0.9/data.yaml"

# Define the dataset paths (you can adjust these paths)
dataset_root_path = "yolo_labels_original_plant_leaf_disease_vit_0.1-0.9"  # Root path to the dataset
train_images_path = "yolo_labels_original_plant_leaf_disease_vit_0.1-0.9/train"  # Relative path to train images
val_images_path = "yolo_labels_original_plant_leaf_disease_vit_0.1-0.9/val"  # Relative path to validation images
test_images_path = "yolo_labels_original_plant_leaf_disease_vit_0.1-0.9/test"  # Optional: specify path if you have test images

# Read class names from label.names
with open(label_names_file, "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Generate the content for the data.yaml file
yaml_content = f"""# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {dataset_root_path}  # dataset root dir
train: {train_images_path}  # train images (relative to 'path')
val: {val_images_path}  # val images (relative to 'path')
test: {test_images_path}  # test images (optional)

# Classes ({len(class_names)} total)
names:"""

# Append class names in the correct format
for idx, class_name in enumerate(class_names):
    yaml_content += f"\n  {idx}: {class_name}"

# Write the yaml content to the output file
with open(output_yaml_file, "w") as yaml_file:
    yaml_file.write(yaml_content)

print(f"data.yaml file has been generated and saved to: {output_yaml_file}")
