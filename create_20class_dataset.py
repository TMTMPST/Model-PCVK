"""
Script to create 20-class dataset from 111 classes
Criteria:
1. Popular batik motifs (well-known)
2. Visually distinct patterns (good for SVM)
3. Geographic diversity (various regions)
"""

import os
import shutil
from pathlib import Path

# 20 Selected classes - visually distinct and popular
SELECTED_CLASSES = [
    # YOGYAKARTA - Famous geometric patterns (very distinct)
    'Yogyakarta_Kawung',            # Circular geometric - ICONIC
    'Yogyakarta_ParangRusak',       # Diagonal sword pattern - ICONIC
    'Yogyakarta_CeplokLiring',      # Symmetrical pattern - distinct
    'Yogyakarta_ParangBarong',      # Large parang - different from ParangRusak

    # JAWA TENGAH - Various distinct patterns
    'JawaTengah_Semarangan',        # Dense floral - very different
    'JawaTengah_Truntum',           # Star pattern - POPULAR
    'JawaTengah_Sidoluhur',         # Wavy pattern - distinct
    'JawaTengah_ParangKusumo',      # Parang variant - different angle
    'JawaTengah_LurikSemangka',     # Stripe pattern - very unique

    # JAWA BARAT - Cloud motif (very distinct)
    'JawaBarat_Megamendung',        # Cloud pattern - ICONIC & UNIQUE

    # JAKARTA - Cultural icon
    'Jakarta_OndelOndel',           # Puppet figure - very distinct

    # BALI - Hindu influence (animal motifs - distinct)
    'Bali_Barong',                  # Mythical creature - UNIQUE
    'Bali_Merak',                   # Peacock - animal motif

    # PAPUA - Tribal geometric (extremely distinct)
    'Papua_Asmat',                  # Tribal pattern - VERY UNIQUE
    'Papua_Cendrawasih',            # Bird of paradise - distinct
    'Papua_Tifa',                   # Drum pattern - distinct

    # KALIMANTAN - Dayak patterns (distinct tribal)
    'Kalimantan_Dayak',             # Tribal geometric - unique

    # LAMPUNG - Unique motif
    'Lampung_Gajah',                # Elephant - animal motif, distinct

    # SULAWESI - Script pattern (very unique)
    'SulawesiSelatan_Lontara',      # Ancient script - VERY DISTINCT
]

# Source and destination directories
SOURCE_DIR = "dataset"  # Use 'dataset' folder which has train/val/test structure
DEST_DIR = "dataset_20class"

def create_20class_dataset():
    """Create 20-class dataset by copying selected classes"""

    print("="*70)
    print("Creating 20-Class Batik Dataset")
    print("="*70)
    print(f"\nSource: {SOURCE_DIR}/")
    print(f"Destination: {DEST_DIR}/\n")

    # Create destination directory
    os.makedirs(DEST_DIR, exist_ok=True)

    # Create train/val/test structure
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(DEST_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

    # Copy each selected class
    copied_classes = []
    missing_classes = []

    for class_name in SELECTED_CLASSES:
        # Copy from each split (train/val/test)
        class_total_images = 0
        class_found = False

        for split in ['train', 'val', 'test']:
            source_split = os.path.join(SOURCE_DIR, split, class_name)
            dest_split = os.path.join(DEST_DIR, split, class_name)

            if os.path.exists(source_split):
                class_found = True
                # Copy entire class folder
                shutil.copytree(source_split, dest_split, dirs_exist_ok=True)

                # Count images
                image_count = len([f for f in os.listdir(dest_split)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_total_images += image_count

        if not class_found:
            print(f"❌ MISSING: {class_name}")
            missing_classes.append(class_name)
            continue

        copied_classes.append(class_name)
        print(f"✅ {class_name} ({class_total_images} images total)")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Total classes selected: {len(SELECTED_CLASSES)}")
    print(f"Successfully copied: {len(copied_classes)}")
    print(f"Missing classes: {len(missing_classes)}")

    if missing_classes:
        print("\n⚠️  Missing classes:")
        for cls in missing_classes:
            print(f"   - {cls}")

    print("\n✅ Dataset created successfully!")
    print(f"Location: {os.path.abspath(DEST_DIR)}/")
    print("\nClass distribution by region:")

    regions = {}
    for cls in copied_classes:
        region = cls.split('_')[0]
        regions[region] = regions.get(region, 0) + 1

    for region, count in sorted(regions.items()):
        print(f"  {region}: {count} classes")

    print("\n" + "="*70)
    print("Next steps:")
    print("="*70)
    print("1. Update notebook: DATASET_ROOT = 'data_20class'")
    print("2. Run notebook with 20 classes")
    print("3. Expected better accuracy with visually distinct classes!")
    print("="*70)

if __name__ == "__main__":
    create_20class_dataset()
