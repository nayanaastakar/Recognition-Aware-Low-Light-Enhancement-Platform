import os
import zipfile
import shutil

PROJECT_ROOT = r"C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement"
DATASETS_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# 1. Setup .gitignore check
gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
with open(gitignore_path, "r", encoding="utf-8") as f:
    gi_content = f.read()

if "data/datasets" not in gi_content:
    with open(gitignore_path, "a", encoding="utf-8") as f:
        f.write("\n# Large Datasets\ndata/datasets/\n*.zip\n")
    print("Added data/datasets/ to .gitignore")

# 2. Extract LOLdataset.zip
lol_zip = r"C:\Users\HP\Downloads\LOLdataset.zip"
lol_target = os.path.join(DATASETS_DIR, "LOL")
if os.path.exists(lol_zip):
    if not os.path.exists(lol_target) or len(os.listdir(lol_target)) == 0:
        print(f"Extracting {lol_zip} to {lol_target}...")
        os.makedirs(lol_target, exist_ok=True)
        with zipfile.ZipFile(lol_zip, 'r') as zip_ref:
            zip_ref.extractall(lol_target)
        print("LOL extraction complete.")
    else:
        print(f"LOL directory {lol_target} already exists with content.")
else:
    print(f"LOL zip {lol_zip} not found.")

# 3. Extract ExDark annotations & images
exdark_target = os.path.join(DATASETS_DIR, "ExDark")
os.makedirs(exdark_target, exist_ok=True)

# Copy/Extract ExDark annotations
exdark_ann_zip = r"C:\Users\HP\Downloads\ExDark_Annno.zip"
ann_target = os.path.join(exdark_target, "annotations")
if os.path.exists(exdark_ann_zip):
    if not os.path.exists(ann_target) or len(os.listdir(ann_target)) == 0:
        print(f"Extracting {exdark_ann_zip} to {ann_target}...")
        os.makedirs(ann_target, exist_ok=True)
        with zipfile.ZipFile(exdark_ann_zip, 'r') as zip_ref:
            zip_ref.extractall(ann_target)
        print("ExDark annotations extracted.")
    else:
        print(f"ExDark annotations directory {ann_target} already exists.")

# Extract ExDark images
exdark_img_zip = r"C:\Users\HP\Downloads\ExDark.zip"
images_target = os.path.join(exdark_target, "images")
if os.path.exists(exdark_img_zip):
    if not os.path.exists(images_target) or len(os.listdir(images_target)) == 0:
        print(f"Extracting {exdark_img_zip} to {images_target}...")
        os.makedirs(images_target, exist_ok=True)
        with zipfile.ZipFile(exdark_img_zip, 'r') as zip_ref:
            zip_ref.extractall(images_target)
        print("ExDark images extracted.")
    else:
        print(f"ExDark images directory {images_target} already exists.")
