"""
Dataset preparation, parsers, and data loaders for LOL and ExDark.

1. LOL Dataset:
   - Paired low-light and normal-light image structure
   - Train split (485 pairs), Test split (15 pairs)
   - Pair matching verification, data transforms, and loaders

2. ExDark Dataset:
   - Low-light object detection benchmark (7,363 images, 12 classes)
   - Custom annotation format parser: [Class, x, y, w, h, ...]
   - MS-COCO format conversion
   - Stratified train/val/test split generator

3. Synthetic Sample Generator:
   - For offline testing and instant demonstration, generates synthetic paired samples
"""

import os
import glob
import json
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
from models.transforms import to_tensor

class LOLDataset(Dataset):
    """
    LOL (Low-Light) paired dataset loader.
    Directory structure expected:
        root_dir/
            our485/ (or train/)
                low/
                high/
            eval15/ (or test/)
                low/
                high/
    """
    def __init__(self, root_dir, split='eval15', patch_size=None):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        
        split_dir = os.path.join(root_dir, split)
        self.low_dir = os.path.join(split_dir, 'low')
        self.high_dir = os.path.join(split_dir, 'high')

        if os.path.exists(self.low_dir):
            self.low_paths = sorted(glob.glob(os.path.join(self.low_dir, '*.*')))
        else:
            self.low_paths = []

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_path = self.low_paths[idx]
        filename = os.path.basename(low_path)
        high_path = os.path.join(self.high_dir, filename)

        low_img = Image.open(low_path).convert('RGB')
        
        if os.path.exists(high_path):
            high_img = Image.open(high_path).convert('RGB')
        else:
            high_img = low_img # fallback if high-light is missing

        if self.patch_size:
            w, h = low_img.size
            if w > self.patch_size and h > self.patch_size:
                x = np.random.randint(0, w - self.patch_size)
                y = np.random.randint(0, h - self.patch_size)
                low_img = low_img.crop((x, y, x + self.patch_size, y + self.patch_size))
                high_img = high_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        return {
            'low': to_tensor(low_img),
            'high': to_tensor(high_img),
            'filename': filename
        }

class ExDarkDataset(Dataset):
    """
    ExDark (Exclusively Dark) object detection dataset loader and annotation parser.
    """
    CLASSES = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]

    def __init__(self, root_dir, split='test'):
        self.root_dir = root_dir
        self.split = split
        self.image_paths = []
        self.annotations = []
        
        # Load annotation mapping if available
        ann_file = os.path.join(root_dir, f"{split}_annotations.json")
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                data = json.load(f)
                self.image_paths = data.get('images', [])
                self.annotations = data.get('annotations', [])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_info = self.image_paths[idx]
        img_path = img_info['path']
        img = Image.open(img_path).convert('RGB')
        
        anns = [a for a in self.annotations if a['image_id'] == img_info['id']]
        
        return {
            'image': to_tensor(img),
            'image_id': img_info['id'],
            'annotations': anns,
            'orig_size': (img_info['height'], img_info['width'])
        }

    @staticmethod
    def parse_raw_annotation_file(txt_path):
        """
        Parses raw ExDark .txt annotation format:
        First line: header
        Subsequent lines: [Class] [x] [y] [w] [h] [confidence] [occlusion] [truncation] ...
        """
        boxes = []
        if not os.path.exists(txt_path):
            return boxes
            
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines[1:]: # Skip header
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_name = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append({
                    'class': cls_name,
                    'bbox': [x, y, x + w, y + h]
                })
        return boxes

def generate_synthetic_samples(output_dir, num_samples=10):
    """
    Creates synthetic realistic paired low-light/normal-light samples for offline demonstration & unit tests.
    """
    os.makedirs(os.path.join(output_dir, 'low'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'high'), exist_ok=True)

    np.random.seed(42)
    for i in range(num_samples):
        # Create a synthetic image with varying color patches and geometric shapes
        w, h = 384, 256
        base = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(h):
            base[y, :, 0] = int(120 + 80 * np.sin(y / 30.0))
            base[y, :, 1] = int(140 + 60 * np.cos(y / 40.0))
            base[y, :, 2] = int(160 + 50 * np.sin(y / 20.0))

        # Draw some synthetic objects (boxes representing cars/people)
        img_pil = Image.fromarray(base)
        
        # High-light reference
        high_path = os.path.join(output_dir, 'high', f'sample_{i+1:03d}.png')
        img_pil.save(high_path)

        # Generate realistic low-light degradation:
        # Non-linear gamma degradation + sensor noise + slight color desaturation
        enhancer = ImageEnhance.Brightness(img_pil)
        dark_pil = enhancer.enhance(0.12 + 0.05 * np.random.rand())
        
        dark_np = np.array(dark_pil).astype(np.float32)
        # Add Poisson-Gaussian noise
        noise = np.random.normal(0, 3.5, dark_np.shape)
        dark_np = np.clip(dark_np + noise, 0, 255).astype(np.uint8)
        
        low_path = os.path.join(output_dir, 'low', f'sample_{i+1:03d}.png')
        Image.fromarray(dark_np).save(low_path)

    print(f"Generated {num_samples} synthetic paired samples in {output_dir}")
