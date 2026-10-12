"""
prepare_yolo_malaria.py
-----------------------
Prepares the YOLO object detection dataset for the SmartCell project.
Converts polygon annotations from the NIH-NLM Thin Blood Smears Pf dataset
(Polygon Set) into standard normalized YOLO bounding-box format.

Dataset characteristics:
- Source: NIH-NLM Thin Blood Smears Pf (Polygon Set)
- Patients: 33 patients, 5 images per patient = 165 total microscope fields
- Image dimensions: 5312 x 2988 pixels
- Classes:
    0: Uninfected RBC
    1: Parasitized RBC
    2: WBC (White_Blood_Cell)
- Split strategy: Patient-level stratified split (Seed 42) to completely prevent data leakage
    Train: 23 patients (115 images, 69.7%)
    Val:    5 patients (25 images,  15.2%)
    Test:   5 patients (25 images,  15.2%)
"""

import os
import shutil
import random
import json
from collections import defaultdict
from PIL import Image

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
SOURCE_POLYGON_DIR = r"D:\smartcell\Polygon Set"
OUTPUT_YOLO_DIR = r"D:\smartcell\yolo_malaria"
REPORT_JSON_PATH = r"D:\smartcell\results\yolo_preparation_report.json"
REPORT_TXT_PATH = r"D:\smartcell\results\yolo_preparation_report.txt"

RANDOM_SEED = 42

CLASS_MAPPING = {
    "Uninfected": 0,
    "Parasitized": 1,
    "White_Blood_Cell": 2
}

CLASS_NAMES = {
    0: "Uninfected RBC",
    1: "Parasitized RBC",
    2: "WBC"
}


def parse_gt_file(gt_path, expected_w=5312, expected_h=2988):
    """
    Parses a single GT annotation text file safely.
    Returns:
        header_info: dict with header metadata
        valid_boxes: list of (class_id, x_center, y_center, width, height, raw_label, bbox_pixels)
        warnings: list of warning strings
    """
    warnings = []
    valid_boxes = []

    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        warnings.append(f"Empty annotation file: {gt_path}")
        return None, valid_boxes, warnings

    header_parts = lines[0].split(",")
    try:
        total_cells_reported = int(header_parts[0].strip())
        img_w = int(header_parts[1].strip())
        img_h = int(header_parts[2].strip())
    except (IndexError, ValueError) as e:
        warnings.append(f"Malformed header in {gt_path}: '{lines[0]}' ({e})")
        img_w, img_h = expected_w, expected_h
        total_cells_reported = len(lines) - 1

    header_info = {
        "reported_cells": total_cells_reported,
        "width": img_w,
        "height": img_h
    }

    for line_idx, line in enumerate(lines[1:], start=2):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Too few tokens ({len(parts)}): '{line}'")
            continue

        cell_id = parts[0]
        raw_label = parts[1]
        comment = parts[2]
        shape = parts[3]

        if raw_label not in CLASS_MAPPING:
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Unknown label '{raw_label}'")
            continue

        class_id = CLASS_MAPPING[raw_label]

        if shape != "Polygon":
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Non-polygon shape '{shape}'")
            continue

        try:
            n_pts = int(parts[4])
        except ValueError:
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Invalid point count '{parts[4]}'")
            continue

        coord_tokens = parts[5:]
        if len(coord_tokens) != 2 * n_pts:
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Coords count mismatch (n_pts={n_pts}, tokens={len(coord_tokens)})")
            continue

        try:
            coords = [float(c) for c in coord_tokens]
        except ValueError as e:
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Coordinate conversion error: {e}")
            continue

        xs = coords[0::2]
        ys = coords[1::2]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        bw = xmax - xmin
        bh = ymax - ymin

        # Validate bounding box
        if bw <= 0 or bh <= 0:
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Degenerate box w={bw}, h={bh}")
            continue

        if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Out-of-bounds box: [{xmin}, {ymin}, {xmax}, {ymax}] for image size [{img_w}, {img_h}]")
            # If coordinates exceed slightly due to floating point, clip within image
            xmin = max(0.0, min(float(img_w), xmin))
            ymin = max(0.0, min(float(img_h), ymin))
            xmax = max(0.0, min(float(img_w), xmax))
            ymax = max(0.0, min(float(img_h), ymax))
            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue

        # Normalized YOLO format (x_center, y_center, width, height)
        x_center = ((xmin + xmax) / 2.0) / float(img_w)
        y_center = ((ymin + ymax) / 2.0) / float(img_h)
        norm_w = bw / float(img_w)
        norm_h = bh / float(img_h)

        # Sanity check normalized coordinates
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < norm_w <= 1.0 and 0.0 < norm_h <= 1.0):
            warnings.append(f"Line {line_idx} in {os.path.basename(gt_path)}: Invalid normalized box: {x_center}, {y_center}, {norm_w}, {norm_h}")
            continue

        valid_boxes.append((class_id, x_center, y_center, norm_w, norm_h, raw_label, (xmin, ymin, xmax, ymax)))

    return header_info, valid_boxes, warnings


def build_patient_split(patient_list, patient_cell_counts, seed=RANDOM_SEED):
    """
    Performs patient-level stratified split (70% train, 15% val, 15% test).
    Patients are stratified into:
    - High parasitemia (>= 50 parasitized cells)
    - Low parasitemia (1 - 49 parasitized cells)
    - Healthy controls (0 parasitized cells)
    """
    high = [p for p in patient_list if patient_cell_counts[p]["Parasitized"] >= 50]
    low = [p for p in patient_list if 0 < patient_cell_counts[p]["Parasitized"] < 50]
    ctrl = [p for p in patient_list if patient_cell_counts[p]["Parasitized"] == 0]

    rng = random.Random(seed)
    rng.shuffle(high)
    rng.shuffle(low)
    rng.shuffle(ctrl)

    # 33 patients total: 23 train (70%), 5 val (15%), 5 test (15%)
    # High: 5 total -> 3 train, 1 val, 1 test
    # Low: 19 total -> 13 train, 3 val, 3 test
    # Ctrl: 9 total -> 7 train, 1 val, 1 test
    train_patients = high[:3] + low[:13] + ctrl[:7]
    val_patients = high[3:4] + low[13:16] + ctrl[7:8]
    test_patients = high[4:5] + low[16:19] + ctrl[8:9]

    patient_to_split = {}
    for p in train_patients:
        patient_to_split[p] = "train"
    for p in val_patients:
        patient_to_split[p] = "val"
    for p in test_patients:
        patient_to_split[p] = "test"

    split_groups = {
        "train": sorted(train_patients),
        "val": sorted(val_patients),
        "test": sorted(test_patients)
    }

    return patient_to_split, split_groups


def main():
    print("=" * 70)
    print("SmartCell: YOLO Malaria Dataset Preparation")
    print("=" * 70)

    # Check source directory
    if not os.path.exists(SOURCE_POLYGON_DIR):
        raise FileNotFoundError(f"Source Polygon Set not found at: {SOURCE_POLYGON_DIR}")

    patients = sorted([p for p in os.listdir(SOURCE_POLYGON_DIR) if os.path.isdir(os.path.join(SOURCE_POLYGON_DIR, p))])
    print(f"Found {len(patients)} patient folders in {SOURCE_POLYGON_DIR}")

    # Step 1: Scan and pair all images with GT files, count labels per patient
    print("\nScanning images and annotation files...")
    patient_files = defaultdict(list)
    patient_label_counts = defaultdict(lambda: {"Uninfected": 0, "Parasitized": 0, "White_Blood_Cell": 0})
    all_warnings = []

    for p in patients:
        p_dir = os.path.join(SOURCE_POLYGON_DIR, p)
        img_dir = os.path.join(p_dir, "Img")
        gt_dir = os.path.join(p_dir, "GT")

        if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
            all_warnings.append(f"Missing Img or GT folder in patient {p}")
            continue

        imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
        gts = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.lower().endswith(".txt")}

        for stem in sorted(imgs.keys()):
            if stem not in gts:
                all_warnings.append(f"Patient {p}: Image {imgs[stem]} has no matching GT file")
                continue
            img_path = os.path.join(img_dir, imgs[stem])
            gt_path = os.path.join(gt_dir, gts[stem])
            patient_files[p].append({
                "stem": stem,
                "img_name": imgs[stem],
                "img_path": img_path,
                "gt_path": gt_path,
                "patient": p
            })

            # Quick label count scan
            with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[1] in patient_label_counts[p]:
                        patient_label_counts[p][parts[1]] += 1

    total_paired_images = sum(len(flist) for flist in patient_files.values())
    print(f"Total paired images found: {total_paired_images} across {len(patient_files)} patients")

    # Step 2: Patient-level stratified split
    print(f"\nComputing patient-level stratified split (Random Seed={RANDOM_SEED})...")
    patient_to_split, split_groups = build_patient_split(patients, patient_label_counts, seed=RANDOM_SEED)

    print(f"Train patients: {len(split_groups['train'])} ({len(split_groups['train']) * 5} images)")
    print(f"Val patients:   {len(split_groups['val'])} ({len(split_groups['val']) * 5} images)")
    print(f"Test patients:  {len(split_groups['test'])} ({len(split_groups['test']) * 5} images)")

    # Step 3: Setup output directories
    os.makedirs(OUTPUT_YOLO_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_YOLO_DIR, "labels", split), exist_ok=True)

    # Step 4: Process images and annotations
    print("\nProcessing and converting annotations to YOLO format...")
    split_stats = {
        "train": {"images": 0, "annotations": 0, "by_class": defaultdict(int), "patients": set()},
        "val":   {"images": 0, "annotations": 0, "by_class": defaultdict(int), "patients": set()},
        "test":  {"images": 0, "annotations": 0, "by_class": defaultdict(int), "patients": set()}
    }

    per_image_manifest = []

    for p in patients:
        split = patient_to_split[p]
        dest_img_dir = os.path.join(OUTPUT_YOLO_DIR, "images", split)
        dest_lbl_dir = os.path.join(OUTPUT_YOLO_DIR, "labels", split)

        for item in patient_files[p]:
            img_path = item["img_path"]
            gt_path = item["gt_path"]
            stem = item["stem"]
            img_name = item["img_name"]

            # Validate actual image dimensions
            with Image.open(img_path) as im:
                actual_w, actual_h = im.size

            header_info, valid_boxes, parse_warnings = parse_gt_file(gt_path, expected_w=actual_w, expected_h=actual_h)
            if parse_warnings:
                all_warnings.extend(parse_warnings)

            # Copy image to YOLO directory
            dest_img_path = os.path.join(dest_img_dir, img_name)
            shutil.copy2(img_path, dest_img_path)

            # Write YOLO label file
            dest_lbl_path = os.path.join(dest_lbl_dir, f"{stem}.txt")
            with open(dest_lbl_path, "w", encoding="utf-8") as lf:
                for box in valid_boxes:
                    cls_id, xc, yc, nw, nh, raw_lbl, _ = box
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

            # Update stats
            split_stats[split]["images"] += 1
            split_stats[split]["annotations"] += len(valid_boxes)
            split_stats[split]["patients"].add(p)
            for box in valid_boxes:
                cls_id = box[0]
                split_stats[split]["by_class"][CLASS_NAMES[cls_id]] += 1

            per_image_manifest.append({
                "filename": img_name,
                "stem": stem,
                "patient": p,
                "split": split,
                "cell_count": len(valid_boxes),
                "uninfected": sum(1 for b in valid_boxes if b[0] == 0),
                "parasitized": sum(1 for b in valid_boxes if b[0] == 1),
                "wbc": sum(1 for b in valid_boxes if b[0] == 2)
            })

    # Step 5: Generate data.yaml
    print("\nGenerating YOLO configuration file (data.yaml)...")
    yaml_path = os.path.join(OUTPUT_YOLO_DIR, "data.yaml")
    yaml_content = (
        f"# SmartCell YOLO Malaria Dataset Configuration\n"
        f"# Source: NIH-NLM Thin Blood Smears Pf (Polygon Set)\n"
        f"path: D:/smartcell/yolo_malaria\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: Uninfected RBC\n"
        f"  1: Parasitized RBC\n"
        f"  2: WBC\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)
    print(f"Saved: {yaml_path}")

    # Step 6: Save Preparation Report
    print("\nSaving preparation report...")
    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)

    report_data = {
        "dataset_name": "NIH-NLM-ThinBloodSmearsPf (Polygon Set)",
        "output_directory": OUTPUT_YOLO_DIR,
        "random_seed": RANDOM_SEED,
        "total_source_patients": len(patients),
        "total_source_images": total_paired_images,
        "classes": CLASS_NAMES,
        "splits": {
            split: {
                "patients_count": len(split_stats[split]["patients"]),
                "patients_list": sorted(list(split_stats[split]["patients"])),
                "images_count": split_stats[split]["images"],
                "total_annotations": split_stats[split]["annotations"],
                "annotations_by_class": dict(split_stats[split]["by_class"])
            }
            for split in ["train", "val", "test"]
        },
        "summary": {
            "total_images": sum(split_stats[s]["images"] for s in ["train", "val", "test"]),
            "total_annotations": sum(split_stats[s]["annotations"] for s in ["train", "val", "test"]),
            "total_by_class": {
                c_name: sum(split_stats[s]["by_class"][c_name] for s in ["train", "val", "test"])
                for c_name in CLASS_NAMES.values()
            },
            "invalid_annotations_skipped": len(all_warnings),
            "warnings": all_warnings
        }
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(report_data, jf, indent=2)

    # Human readable text report
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as tf:
        tf.write("======================================================================\n")
        tf.write("SMARTCELL: YOLO MALARIA DATASET PREPARATION REPORT\n")
        tf.write("======================================================================\n\n")
        tf.write(f"Source: NIH-NLM Thin Blood Smears Pf (Polygon Set)\n")
        tf.write(f"Output Directory: {OUTPUT_YOLO_DIR}\n")
        tf.write(f"Random Seed: {RANDOM_SEED}\n")
        tf.write(f"Total Patients: {len(patients)}\n")
        tf.write(f"Total Images: {total_paired_images} (5312 x 2988 px)\n")
        tf.write(f"Total Annotations: {report_data['summary']['total_annotations']}\n")
        tf.write(f"Invalid / Skipped Annotations: {len(all_warnings)}\n\n")

        tf.write("Class Mapping:\n")
        for cid, cname in CLASS_NAMES.items():
            tf.write(f"  Class {cid}: {cname} (Total: {report_data['summary']['total_by_class'][cname]:,})\n")
        tf.write("\n")

        tf.write("Dataset Split Breakdown (Patient-Level Stratification):\n")
        tf.write("-" * 70 + "\n")
        for s in ["train", "val", "test"]:
            tf.write(f"[{s.upper()}]\n")
            tf.write(f"  Patients: {len(split_stats[s]['patients'])}\n")
            tf.write(f"  Images: {split_stats[s]['images']}\n")
            tf.write(f"  Total Cells: {split_stats[s]['annotations']:,}\n")
            for cname in CLASS_NAMES.values():
                c_count = split_stats[s]["by_class"][cname]
                pct = (c_count / split_stats[s]["annotations"] * 100) if split_stats[s]["annotations"] > 0 else 0
                tf.write(f"    - {cname}: {c_count:,} ({pct:.2f}%)\n")
            tf.write("\n")

        tf.write("-" * 70 + "\n")
        tf.write("Patient Allocation:\n")
        for s in ["train", "val", "test"]:
            tf.write(f"  {s.upper()}: {', '.join(sorted(list(split_stats[s]['patients'])))}\n\n")

        tf.write("-" * 70 + "\n")
        tf.write(f"Warnings / Invalid Annotations: {len(all_warnings)}\n")
        for w in all_warnings[:20]:
            tf.write(f"  WARNING: {w}\n")
        if len(all_warnings) > 20:
            tf.write(f"  ... and {len(all_warnings) - 20} more warnings.\n")

    print(f"Saved: {REPORT_JSON_PATH}")
    print(f"Saved: {REPORT_TXT_PATH}")

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total Images:      {report_data['summary']['total_images']}")
    print(f"Total Annotations: {report_data['summary']['total_annotations']}")
    print(f"Train Annotations: {split_stats['train']['annotations']} (Images: {split_stats['train']['images']})")
    print(f"Val Annotations:   {split_stats['val']['annotations']} (Images: {split_stats['val']['images']})")
    print(f"Test Annotations:  {split_stats['test']['annotations']} (Images: {split_stats['test']['images']})")
    print(f"Invalid / Skipped: {len(all_warnings)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
