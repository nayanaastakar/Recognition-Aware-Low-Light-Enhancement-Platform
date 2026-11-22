import os
import re
import json
import xml.etree.ElementTree as ET
import numpy as np

# Official ExDark 12 classes mapping
EXDARK_CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]
EXDARK_LOWER = [c.lower() for c in EXDARK_CLASSES]

def parse_annotation_content(content_str, img_width, img_height, filename=""):
    """
    Parses ExDark annotation from string content.
    Supports:
    1. ExDark official space-separated text format:
       <class_name> <x_min> <y_min> <width> <height> ...
       or <class_name> <x_min> <y_min> <x_max> <y_max> ...
    2. YOLO normalized format:
       <class_id> <x_center> <y_center> <width> <height>
    3. Pascal VOC XML format:
       <annotation><object><name>...</name><bndbox>...</bndbox></object></annotation>
    4. COCO JSON format:
       {"images": [...], "annotations": [...], "categories": [...]}
       
    Returns:
        list of dicts: [{'class': str, 'bbox': [x1, y1, x2, y2]}]
    """
    gts = []
    text = content_str.strip()
    if not text:
        return gts

    # 1. Try JSON (COCO or custom)
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Standard COCO
                if "annotations" in data:
                    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
                    for ann in data["annotations"]:
                        cls_name = cat_map.get(ann.get("category_id"), "object").lower()
                        bbox = ann.get("bbox", [])
                        if len(bbox) == 4:
                            x, y, w, h = bbox
                            gts.append({'class': cls_name, 'bbox': [int(x), int(y), int(x + w), int(y + h)]})
                    return gts
                # List of boxes under key
                for key in ["objects", "boxes", "detections", "annotations"]:
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            cls_name = str(item.get("class", item.get("name", item.get("label", "object")))).lower()
                            box = item.get("box", item.get("bbox", []))
                            if len(box) == 4:
                                gts.append({'class': cls_name, 'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])]})
                        return gts
        except Exception:
            pass

    # 2. Try Pascal VOC XML
    if "<annotation>" in text or "<object>" in text:
        try:
            root = ET.fromstring(text)
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                cls_name = name_elem.text.lower() if name_elem is not None else "object"
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    gts.append({'class': cls_name, 'bbox': [xmin, ymin, xmax, ymax]})
            return gts
        except Exception:
            pass

    # 3. Try Text line-by-line (ExDark official or YOLO)
    # Replace commas with spaces in case of CSV-style text
    cleaned_text = text.replace(",", " ")
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("%") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        # Case A: First element is numeric (YOLO format: class_id x_center y_center w h)
        if re.match(r'^\d+$', parts[0]) and len(parts) == 5:
            try:
                cls_id = int(parts[0])
                cls_name = EXDARK_CLASSES[cls_id].lower() if cls_id < len(EXDARK_CLASSES) else f"class_{cls_id}"
                xc = float(parts[1]) * img_width
                yc = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
                x1 = int(max(0, xc - w / 2))
                y1 = int(max(0, yc - h / 2))
                x2 = int(min(img_width, xc + w / 2))
                y2 = int(min(img_height, yc + h / 2))
                gts.append({'class': cls_name, 'bbox': [x1, y1, x2, y2]})
                continue
            except Exception:
                pass

        # Case B: ExDark official format:
        # Format: <class_name> <x_min> <y_min> <width> <height> ...
        cls_candidate = parts[0].lower()
        try:
            val1 = float(parts[1])
            val2 = float(parts[2])
            val3 = float(parts[3])
            val4 = float(parts[4])

            # Check if coordinates are normalized (0.0 to 1.0)
            if val1 <= 1.0 and val2 <= 1.0 and val3 <= 1.0 and val4 <= 1.0:
                val1 *= img_width
                val2 *= img_height
                val3 *= img_width
                val4 *= img_height

            # In official ExDark dataset annotations:
            # format is: <class> <x_min> <y_min> <width> <height> ...
            # or if already absolute: <class> <xmin> <ymin> <xmax> <ymax>
            if val3 > val1 and val4 > val2 and (val3 <= img_width) and (val4 <= img_height) and (val1 + val3 > img_width):
                # Unambiguously xmin, ymin, xmax, ymax
                x1 = int(val1)
                y1 = int(val2)
                x2 = int(val3)
                y2 = int(val4)
            else:
                # ExDark default: x_min, y_min, width, height
                x1 = int(val1)
                y1 = int(val2)
                x2 = int(val1 + val3)
                y2 = int(val2 + val4)

            # Bound to image dimensions
            x1 = max(0, min(img_width, x1))
            y1 = max(0, min(img_height, y1))
            x2 = max(0, min(img_width, x2))
            y2 = max(0, min(img_height, y2))

            if x2 > x1 and y2 > y1:
                gts.append({'class': cls_candidate, 'bbox': [x1, y1, x2, y2]})
        except Exception:
            continue

    return gts
