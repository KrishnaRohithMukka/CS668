import os
import shutil

# Define the base directory containing all datasets
base_dir = r"C:\Users\92472\Downloads\ANOMOALY"  # Update with your path to the folder containing Vandalism, Violence, Assault, etc.
target_dir = r"C:\Users\92472\Downloads\ANOMOALY\data_set"  # Update with your desired output directory for the merged dataset

# Define train and val target directories within the merged dataset
train_image_target = os.path.join(target_dir, "train", "images")
train_label_target = os.path.join(target_dir, "train", "labels")
val_image_target = os.path.join(target_dir, "val", "images")
val_label_target = os.path.join(target_dir, "val", "labels")

# Create directories if they don't exist
os.makedirs(train_image_target, exist_ok=True)
os.makedirs(train_label_target, exist_ok=True)
os.makedirs(val_image_target, exist_ok=True)
os.makedirs(val_label_target, exist_ok=True)

# Dataset folders to merge
dataset_folders = ["Vandalism", "Violence", "Assault", "Fire", "Accident"]

def copy_files(src_image_dir, src_label_dir, dest_image_dir, dest_label_dir, prefix):
    # Copy images
    for file_name in os.listdir(src_image_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            src_image_path = os.path.join(src_image_dir, file_name)
            new_image_name = f"{prefix}_{file_name}"
            dest_image_path = os.path.join(dest_image_dir, new_image_name)
            shutil.copy(src_image_path, dest_image_path)
            print(f"Copied image {src_image_path} to {dest_image_path}")
    
    # Copy labels
    for file_name in os.listdir(src_label_dir):
        if file_name.endswith('.txt'):
            src_label_path = os.path.join(src_label_dir, file_name)
            new_label_name = f"{prefix}_{file_name}"
            dest_label_path = os.path.join(dest_label_dir, new_label_name)
            shutil.copy(src_label_path, dest_label_path)
            print(f"Copied label {src_label_path} to {dest_label_path}")

# Iterate over each dataset folder and copy files
for dataset in dataset_folders:
    train_image_dir = os.path.join(base_dir, dataset, "train", "images")
    train_label_dir = os.path.join(base_dir, dataset, "train", "labels")
    val_image_dir = os.path.join(base_dir, dataset, "val", "images")
    val_label_dir = os.path.join(base_dir, dataset, "val", "labels")

    # Use the dataset name as a prefix to avoid filename conflicts
    prefix = dataset.lower()

    # Copy train files
    if os.path.exists(train_image_dir) and os.path.exists(train_label_dir):
        copy_files(train_image_dir, train_label_dir, train_image_target, train_label_target, prefix)

    # Copy val files
    if os.path.exists(val_image_dir) and os.path.exists(val_label_dir):
        copy_files(val_image_dir, val_label_dir, val_image_target, val_label_target, prefix)

print("Merging complete.")


