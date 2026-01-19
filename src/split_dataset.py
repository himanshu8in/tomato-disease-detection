# Dataset splitting script
import os
import shutil
import random

SOURCE_DIR = r"C:\Users\himan\OneDrive\Desktop\Plant_disease\PlantVillage-Dataset\raw\color"
DEST_DIR = r"C:\Users\himan\OneDrive\Desktop\Plant_disease\dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

for split in ["train", "validation", "test"]:
    split_path = os.path.join(DEST_DIR, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)
        
classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".png"))]
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val
    
    for split in ["train","validation", "test"]:
        split_clas_path = os.path.join(DEST_DIR, split, cls)
        os.makedirs(split_clas_path, exist_ok=True)
        
    for i, img in enumerate(images):
        src_img_path = os.path.join(cls_path, img)
        if i < n_train:
            dst_img_path = os.path.join(DEST_DIR, "train", cls, img)
        elif i < n_train + n_val:
            dst_img_path = os.path.join(DEST_DIR, "validation", cls,img)
        else:
            dst_img_path = os.path.join(DEST_DIR,"test", cls,img)
        shutil.copy2(src_img_path, dst_img_path)
        
        print(f"Class '{cls}': total={n_total}, train={n_train}, val={n_val}, test={n_test}")
print("\nDataset split completed")