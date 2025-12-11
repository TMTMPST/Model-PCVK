"""
Script to split dataset into train, validation, and test sets
Proportions: 70% train, 15% validation, 15% test
"""

import os
import shutil
from pathlib import Path
import random
import time
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)

# Paths
SOURCE_DIR = Path('data')
DEST_DIR = Path('dataset')

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("="*80)
print("DATASET SPLITTING TOOL")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Source: {SOURCE_DIR}")
print(f"Destination: {DEST_DIR}")
print(f"Split ratio: Train={TRAIN_RATIO*100}%, Val={VAL_RATIO*100}%, Test={TEST_RATIO*100}%")
print("="*80)

# Start timer
start_time = time.time()

# Create destination directories
for split in ['train', 'val', 'test']:
    split_dir = DEST_DIR / split
    if split_dir.exists():
        print(f"\nWARNING: {split_dir} already exists!")
        response = input(f"Delete and recreate? (yes/no): ")
        if response.lower() == 'yes':
            shutil.rmtree(split_dir)
            print(f"Deleted {split_dir}")
        else:
            print("Aborting. Please backup or rename existing data directory.")
            exit()
    split_dir.mkdir(parents=True, exist_ok=True)

# Get all class folders
class_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]
class_folders = sorted(class_folders)

print(f"\nFound {len(class_folders)} classes")
print("="*80)

total_images = 0
total_train = 0
total_val = 0
total_test = 0

# Process each class
for class_folder in class_folders:
    class_name = class_folder.name

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(class_folder.glob(ext)))

    if len(image_files) == 0:
        print(f"WARNING: No images found in {class_name}")
        continue

    # Shuffle images
    random.shuffle(image_files)

    # Calculate split indices
    n_images = len(image_files)
    n_train = int(n_images * TRAIN_RATIO)
    n_val = int(n_images * VAL_RATIO)
    n_test = n_images - n_train - n_val  # Remaining goes to test

    # Split images
    train_images = image_files[:n_train]
    val_images = image_files[n_train:n_train + n_val]
    test_images = image_files[n_train + n_val:]

    # Create class directories in each split
    for split in ['train', 'val', 'test']:
        (DEST_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    # Copy images to respective directories
    for img in train_images:
        shutil.copy2(img, DEST_DIR / 'train' / class_name / img.name)

    for img in val_images:
        shutil.copy2(img, DEST_DIR / 'val' / class_name / img.name)

    for img in test_images:
        shutil.copy2(img, DEST_DIR / 'test' / class_name / img.name)

    # Update counters
    total_images += n_images
    total_train += n_train
    total_val += n_val
    total_test += n_test

    print(f"{class_name:40s}: {n_images:4d} total -> Train: {n_train:3d}, Val: {n_val:3d}, Test: {n_test:3d}")

# Calculate elapsed time
elapsed_time = time.time() - start_time

print("="*80)
print("SUMMARY")
print("="*80)

if total_images == 0:
    print("❌ No images found to process!")
    print("Please check if the dataset directory exists and contains image files.")
    print("="*80)
    exit(1)

print(f"Total images processed: {total_images}")
print(f"Train: {total_train} ({total_train/total_images*100:.1f}%)")
print(f"Val:   {total_val} ({total_val/total_images*100:.1f}%)")
print(f"Test:  {total_test} ({total_test/total_images*100:.1f}%)")
print("="*80)
print("\n✅ Dataset split completed successfully!")
print(f"\nDirectory structure:")
print(f"  {DEST_DIR}/")
print(f"    train/ ({total_train} images)")
print(f"    val/ ({total_val} images)")
print(f"    test/ ({total_test} images)")
print()
print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"📊 Processing speed: {total_images/elapsed_time:.1f} images/second")
print(f"🏁 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
