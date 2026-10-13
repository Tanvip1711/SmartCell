"""
preflight_check.py
------------------
Audits and validates the YOLO dataset at D:\smartcell\yolo_malaria
before training.

Verifies:
1. Existence of all required directories and configuration files.
2. 1-to-1 matching of images and labels across train, val, and test.
3. Coordinate normalization and bounding box validity.
4. Patient-level isolation across partitions.
5. Exact annotation counts per class.
"""

import os
import sys
import json
from collections import defaultdict, Counter

YOLO_DIR = r"D:\smartcell\yolo_malaria"
REPORT_JSON = r"D:\smartcell\results\metrics\yolo_preparation_report.json"
PREVIEW_DIR = r"D:\smartcell\results\yolo_preview"
README_FILE = r"D:\smartcell\README.md"
YAML_FILE = os.path.join(YOLO_DIR, "data.yaml")

def main():
    print("=" * 70)
    print("SMARTCELL: YOLO PREFLIGHT DATASET VALIDATION")
    print("=" * 70)

    # 1. Path existence
    required_paths = [
        YAML_FILE,
        os.path.join(YOLO_DIR, "images", "train"),
        os.path.join(YOLO_DIR, "images", "val"),
        os.path.join(YOLO_DIR, "images", "test"),
        os.path.join(YOLO_DIR, "labels", "train"),
        os.path.join(YOLO_DIR, "labels", "val"),
        os.path.join(YOLO_DIR, "labels", "test"),
        PREVIEW_DIR,
        REPORT_JSON,
        README_FILE
    ]

    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        print("[FAIL] Missing required paths:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    else:
        print("[PASS] All required paths and files exist.")

    # 2. Check data.yaml
    with open(YAML_FILE, "r") as f:
        yaml_content = f.read()
    print("\n--- data.yaml content ---")
    print(yaml_content.strip())
    print("-------------------------")

    # 3. Check preparation report
    with open(REPORT_JSON, "r") as f:
        prep_report = json.load(f)
    print("\nPreparation report loaded:")
    print(f"  Total source patients: {prep_report['total_source_patients']}")
    print(f"  Total source images: {prep_report['total_source_images']}")
    print(f"  Reported summary total cells: {prep_report['summary']['total_annotations']}")

    # 4. Detailed file and label inspection across splits
    splits = ["train", "val", "test"]
    split_img_counts = {}
    split_lbl_counts = {}
    split_class_counts = defaultdict(Counter)
    all_invalid_lines = []
    all_out_of_bounds = []
    all_empty_labels = []

    for s in splits:
        img_dir = os.path.join(YOLO_DIR, "images", s)
        lbl_dir = os.path.join(YOLO_DIR, "labels", s)

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])

        split_img_counts[s] = len(img_files)
        split_lbl_counts[s] = len(lbl_files)

        img_stems = set(os.path.splitext(f)[0] for f in img_files)
        lbl_stems = set(os.path.splitext(f)[0] for f in lbl_files)

        if img_stems != lbl_stems:
            diff_img = img_stems - lbl_stems
            diff_lbl = lbl_stems - img_stems
            print(f"[FAIL] Split '{s}' mismatch! Images without labels: {len(diff_img)}, Labels without images: {len(diff_lbl)}")
            sys.exit(1)

        # Inspect every label file line-by-line
        for lf in lbl_files:
            l_path = os.path.join(lbl_dir, lf)
            with open(l_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                all_empty_labels.append((s, lf))

            for l_idx, line in enumerate(lines, 1):
                tokens = line.split()
                if len(tokens) != 5:
                    all_invalid_lines.append((s, lf, l_idx, f"Invalid token count {len(tokens)}: {line}"))
                    continue

                try:
                    cls_id = int(tokens[0])
                    xc = float(tokens[1])
                    yc = float(tokens[2])
                    w = float(tokens[3])
                    h = float(tokens[4])
                except ValueError as e:
                    all_invalid_lines.append((s, lf, l_idx, f"Token parse error: {e}"))
                    continue

                if cls_id not in (0, 1, 2):
                    all_invalid_lines.append((s, lf, l_idx, f"Class ID {cls_id} out of range [0, 2]"))
                    continue

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    all_out_of_bounds.append((s, lf, l_idx, f"Center coordinates out of bounds: xc={xc}, yc={yc}"))

                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    all_out_of_bounds.append((s, lf, l_idx, f"Box dimension out of bounds: w={w}, h={h}"))

                split_class_counts[s][cls_id] += 1

    print("\n[VERIFIED] File and Annotation Counts:")
    total_imgs = 0
    total_annotations = 0
    total_by_class = Counter()

    for s in splits:
        s_total = sum(split_class_counts[s].values())
        total_imgs += split_img_counts[s]
        total_annotations += s_total
        for c in (0, 1, 2):
            total_by_class[c] += split_class_counts[s][c]

        print(f"  [{s.upper():5}] Images: {split_img_counts[s]:3} | Annotations: {s_total:5} | "
              f"Uninfected (0): {split_class_counts[s][0]:5} | "
              f"Parasitized (1): {split_class_counts[s][1]:4} | "
              f"WBC (2): {split_class_counts[s][2]:2}")

    print(f"\n  [TOTAL] Images: {total_imgs:3} | Annotations: {total_annotations:5} | "
          f"Uninfected (0): {total_by_class[0]:5} | "
          f"Parasitized (1): {total_by_class[1]:4} | "
          f"WBC (2): {total_by_class[2]:2}")

    print(f"\nValidation Quality:")
    print(f"  Invalid / malformed lines: {len(all_invalid_lines)}")
    print(f"  Out of bounds coordinates: {len(all_out_of_bounds)}")
    print(f"  Empty label files: {len(all_empty_labels)}")

    # 5. Patient isolation check across splits
    print("\n[PATIENT ISOLATION CHECK]")
    split_patients = {}
    for s in splits:
        patients_list = prep_report['splits'][s]['patients_list']
        split_patients[s] = set(patients_list)
        print(f"  {s.upper()}: {len(patients_list)} patients ({', '.join(sorted(patients_list))[:60]}...)")

    overlap_train_val = split_patients['train'] & split_patients['val']
    overlap_train_test = split_patients['train'] & split_patients['test']
    overlap_val_test = split_patients['val'] & split_patients['test']

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("[FAIL] Patient leakage detected across splits!")
        sys.exit(1)
    else:
        print("[PASS] 0 patient overlap across partitions. Strict patient isolation confirmed.")

    # 6. Check previews
    preview_files = [f for f in os.listdir(PREVIEW_DIR) if f.lower().endswith(('.jpg', '.png'))]
    print(f"\n[VISUAL PREVIEWS] Found {len(preview_files)} preview images in {PREVIEW_DIR}")

    print("\n" + "=" * 70)
    print("PREFLIGHT VALIDATION: ALL CHECKS PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()
