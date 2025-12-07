#!/usr/bin/env python3
"""
Evaluate YOLOv8 predictions loaded from CSV or YOLO txt files on validation set using mAP@0.5
"""

from pathlib import Path
import os
import csv

# ============================
# CONFIGURATION
# ============================
WORK_DIR = Path(".")
VAL_IMAGES_DIR = WORK_DIR / "val/images"
VAL_LABELS_DIR = WORK_DIR / "val/labels"
PRED_DIR = WORK_DIR / "predictions/labels"  # Folder with YOLO txt predictions
IOU_THRESHOLD = 0.5
CLASS_NAMES = ["Carpetweed", "Morning Glory", "Palmer Amaranth"]

# ============================
# UTILITY FUNCTIONS
# ============================

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa1, ya1 = x1 - w1/2, y1 - h1/2
    xa2, ya2 = x1 + w1/2, y1 + h1/2
    xb1, yb1 = x2 - w2/2, y2 - h2/2
    xb2, yb2 = x2 + w2/2, y2 + h2/2
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def load_gt(label_path):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            c, x, y, w, h = map(float, line.strip().split())
            boxes.append([int(c), x, y, w, h])
    return boxes

def load_pred(pred_path):
    if not os.path.exists(pred_path):
        return []
    boxes = []
    with open(pred_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                c, conf, x, y, w, h = parts
                boxes.append([int(c), float(conf), float(x), float(y), float(w), float(h)])
    return boxes

def compute_ap(gt_boxes, pred_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0:
        return 0.0
    matched = set()
    tp, fp = 0, 0
    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: -x[1])
    for pred in pred_boxes_sorted:
        p_class, p_conf, *p_box = pred
        best_iou, best_idx = 0, -1
        for idx, gt in enumerate(gt_boxes):
            gt_class, *gt_box = gt
            if gt_class != p_class:
                continue
            current_iou = iou(p_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx
        if best_iou >= iou_thresh and best_idx not in matched:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1
    precision = tp / (tp + fp + 1e-9)
    return precision

# ============================
# RUN EVALUATION
# ============================
val_images = list(VAL_IMAGES_DIR.glob("*.jpg"))
aps = []

print(f"Evaluating {len(val_images)} validation images using predictions from '{PRED_DIR}'...\n")

for img_path in val_images:
    image_id = img_path.stem
    gt_boxes = load_gt(VAL_LABELS_DIR / f"{image_id}.txt")
    pred_boxes = load_pred(PRED_DIR / f"{image_id}.txt")
    ap = compute_ap(gt_boxes, pred_boxes, IOU_THRESHOLD)
    aps.append(ap)

map50 = sum(aps) / len(aps)
print("="*50)
print(f"Offline mAP@0.5 on validation set: {map50:.4f}")
print("="*50)
