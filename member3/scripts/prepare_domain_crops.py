r"""
prepare_domain_crops.py
-----------------------
SmartCell Phase 6: Domain-Matched RBC Crop Dataset Preparation

Extracts ground-truth RBC crops from the full-resolution NIH-NLM Thin Blood Smears Pf
microscope images in D:\smartcell\yolo_malaria\images\{train,val,test} using the
ground-truth bounding-box annotations in D:\smartcell\yolo_malaria\labels\{train,val,test}.

Rules Enforced:
1. Strictly preserves the existing patient-level split (23 train / 5 val / 5 test patients).
   Verifies zero patient leakage across partitions.
2. Uses ground-truth annotations (0 = Uninfected, 1 = Parasitized) — never YOLO or CNN predictions.
3. Excludes WBC annotations (class 2) from binary malaria classification and logs all exclusions.
4. Detects and suppresses duplicate polygon annotations (IoU >= 0.80) and records all moderate
   same-class and cross-class overlaps (IoU >= 0.20).
5. Saves two controlled crop variants with pad_px=4 (matching run_smartcell_pipeline.py):
   - raw: Unmodified rectangular RGB crops
   - masked: Elliptical-masked RGB crops with black [0,0,0] corners
6. Outputs a complete CSV manifest and JSON/TXT dataset preparation reports.
"""

import os
import csv
import json
import time
from collections import defaultdict
import numpy as np
from PIL import Image

BASE_DIR = r"D:\smartcell"
POLYGON_SET_DIR = os.path.join(BASE_DIR, "Polygon Set")
YOLO_DIR = os.path.join(BASE_DIR, "yolo_malaria")
YOLO_PREP_REPORT = os.path.join(BASE_DIR, "results", "metrics", "yolo_preparation_report.json")

DOMAIN_ADAPT_DIR = os.path.join(BASE_DIR, "results", "domain_adaptation")
CROPS_DATASET_DIR = os.path.join(DOMAIN_ADAPT_DIR, "crops_dataset")
MANIFEST_CSV_PATH = os.path.join(CROPS_DATASET_DIR, "domain_crops_manifest.csv")
REPORT_JSON_PATH = os.path.join(BASE_DIR, "results", "metrics", "phase6_domain_crops_report.json")
REPORT_TXT_PATH = os.path.join(DOMAIN_ADAPT_DIR, "phase6_domain_crops_report.txt")

PAD_PX = 4
MIN_BOX_SIZE = 16
DUPLICATE_IOU_THRESH = 0.80
AMBIGUOUS_OVERLAP_IOU_THRESH = 0.20

CLASS_MAP = {
    0: "Uninfected",
    1: "Parasitized"
}


def compute_iou(box_a, box_b):
    """Computes IoU between two boxes [x1, y1, x2, y2]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = float((x_right - x_left) * (y_bottom - y_top))
    area_a = float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def apply_elliptical_mask(crop_rgb_np):
    """Zeroes out pixels outside the inscribed ellipse of a rectangular cell crop."""
    ch, cw = crop_rgb_np.shape[:2]
    yy, xx = np.ogrid[:ch, :cw]
    cy, cx = ch / 2.0, cw / 2.0
    ry, rx = ch / 2.0, cw / 2.0
    mask = ((xx - cx) ** 2 / (rx ** 2 + 1e-6) + (yy - cy) ** 2 / (ry ** 2 + 1e-6)) <= 1.0
    masked = crop_rgb_np.copy()
    masked[~mask] = 0
    return masked


def build_image_to_patient_map():
    """Maps each image filename and stem from Polygon Set to its patient ID."""
    img_to_patient = {}
    if not os.path.exists(POLYGON_SET_DIR):
        raise FileNotFoundError(f"Polygon Set directory not found: {POLYGON_SET_DIR}")

    patients = sorted([
        p for p in os.listdir(POLYGON_SET_DIR)
        if os.path.isdir(os.path.join(POLYGON_SET_DIR, p))
    ])
    for p in patients:
        img_dir = os.path.join(POLYGON_SET_DIR, p, "Img")
        if not os.path.exists(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                stem = os.path.splitext(fname)[0]
                img_to_patient[stem] = p
                img_to_patient[fname] = p
    return img_to_patient


def main():
    t0 = time.time()
    print("=" * 78)
    print("SMARTCELL PHASE 6: DOMAIN-MATCHED CROP DATASET PREPARATION")
    print("=" * 78)

    os.makedirs(CROPS_DATASET_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)

    for mode in ["raw", "masked"]:
        for split in ["train", "val", "test"]:
            for cname in ["Uninfected", "Parasitized"]:
                os.makedirs(os.path.join(CROPS_DATASET_DIR, mode, split, cname), exist_ok=True)

    img_to_patient = build_image_to_patient_map()
    with open(YOLO_PREP_REPORT, "r", encoding="utf-8") as f:
        yolo_prep = json.load(f)

    expected_split_patients = {
        split: set(yolo_prep["splits"][split]["patients_list"])
        for split in ["train", "val", "test"]
    }

    # Verify zero patient overlap in expected splits
    assert expected_split_patients["train"].isdisjoint(expected_split_patients["val"])
    assert expected_split_patients["train"].isdisjoint(expected_split_patients["test"])
    assert expected_split_patients["val"].isdisjoint(expected_split_patients["test"])

    manifest_rows = []
    excluded_wbc_records = []
    invalid_box_records = []
    duplicate_suppressed_records = []
    ambiguous_overlap_records = []

    split_summary = {}
    patient_summary = defaultdict(lambda: {
        "partition": "",
        "images_count": 0,
        "Uninfected": 0,
        "Parasitized": 0,
        "excluded_wbc": 0,
        "suppressed_duplicates": 0,
        "ambiguous_overlaps_flagged": 0
    })

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(YOLO_DIR, "images", split)
        lbl_dir = os.path.join(YOLO_DIR, "labels", split)
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        split_patients_seen = set()
        split_counts = {
            "images_count": len(img_files),
            "patients_count": 0,
            "total_raw_annotations": 0,
            "valid_rbc_crops_extracted": 0,
            "Uninfected": 0,
            "Parasitized": 0,
            "excluded_wbc_count": 0,
            "invalid_box_count": 0,
            "duplicate_boxes_suppressed": 0,
            "ambiguous_same_class_overlaps_flagged": 0,
            "ambiguous_cross_class_overlaps_flagged": 0
        }

        print(f"\nProcessing [{split.upper()}] partition ({len(img_files)} microscope images)...")

        for img_idx, fname in enumerate(img_files, start=1):
            stem = os.path.splitext(fname)[0]
            patient_id = img_to_patient.get(stem)
            if patient_id is None:
                raise RuntimeError(f"Could not map image {fname} to a patient ID!")
            if patient_id not in expected_split_patients[split]:
                raise RuntimeError(
                    f"Patient leakage check failed! Image {fname} (patient {patient_id}) "
                    f"is in {split}, not in expected {split} patient list."
                )

            split_patients_seen.add(patient_id)
            patient_summary[patient_id]["partition"] = split
            patient_summary[patient_id]["images_count"] += 1

            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                img_w, img_h = rgb_img.size
                img_np = np.array(rgb_img)

            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = [ln.strip() for ln in lf if ln.strip()]

            split_counts["total_raw_annotations"] += len(lines)

            candidate_rbcs = []
            for line_num, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    invalid_box_records.append({
                        "partition": split,
                        "source_image": fname,
                        "patient_id": patient_id,
                        "line_number": line_num,
                        "reason": f"Malformed YOLO line with {len(parts)} tokens"
                    })
                    split_counts["invalid_box_count"] += 1
                    continue

                cid = int(parts[0])
                xc, yc, nw, nh = map(float, parts[1:5])

                bw_f = nw * img_w
                bh_f = nh * img_h
                x1_f = (xc * img_w) - (bw_f / 2.0)
                y1_f = (yc * img_h) - (bh_f / 2.0)
                x2_f = (xc * img_w) + (bw_f / 2.0)
                y2_f = (yc * img_h) + (bh_f / 2.0)

                x1 = max(0, min(img_w, int(round(x1_f))))
                y1 = max(0, min(img_h, int(round(y1_f))))
                x2 = max(0, min(img_w, int(round(x2_f))))
                y2 = max(0, min(img_h, int(round(y2_f))))
                bw = x2 - x1
                bh = y2 - y1

                if cid == 2:
                    excluded_wbc_records.append({
                        "partition": split,
                        "source_image": fname,
                        "patient_id": patient_id,
                        "line_number": line_num,
                        "bbox": [x1, y1, x2, y2],
                        "reason": "WBC (class 2) excluded from binary malaria RBC dataset"
                    })
                    split_counts["excluded_wbc_count"] += 1
                    patient_summary[patient_id]["excluded_wbc"] += 1
                    continue

                if cid not in (0, 1):
                    invalid_box_records.append({
                        "partition": split,
                        "source_image": fname,
                        "patient_id": patient_id,
                        "line_number": line_num,
                        "reason": f"Unsupported class ID {cid}"
                    })
                    split_counts["invalid_box_count"] += 1
                    continue

                if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
                    invalid_box_records.append({
                        "partition": split,
                        "source_image": fname,
                        "patient_id": patient_id,
                        "line_number": line_num,
                        "bbox": [x1, y1, x2, y2],
                        "reason": f"Box dimensions {bw}x{bh} smaller than minimum {MIN_BOX_SIZE}px"
                    })
                    split_counts["invalid_box_count"] += 1
                    continue

                candidate_rbcs.append({
                    "line_number": line_num,
                    "class_id": cid,
                    "class_label": CLASS_MAP[cid],
                    "bbox": [x1, y1, x2, y2],
                    "bw": bw,
                    "bh": bh
                })

            # Step A: Suppress duplicate annotations (IoU >= 0.80)
            kept_rbcs = []
            for cand in candidate_rbcs:
                is_dup = False
                for existing in kept_rbcs:
                    iou_val = compute_iou(cand["bbox"], existing["bbox"])
                    if iou_val >= DUPLICATE_IOU_THRESH:
                        is_dup = True
                        duplicate_suppressed_records.append({
                            "partition": split,
                            "source_image": fname,
                            "patient_id": patient_id,
                            "suppressed_line": cand["line_number"],
                            "kept_line": existing["line_number"],
                            "class_label": cand["class_label"],
                            "iou": round(iou_val, 4),
                            "bbox": cand["bbox"]
                        })
                        split_counts["duplicate_boxes_suppressed"] += 1
                        patient_summary[patient_id]["suppressed_duplicates"] += 1
                        break
                if not is_dup:
                    kept_rbcs.append(cand)

            # Step B: Compute pairwise overlap metrics & extract raw + masked crops
            n_kept = len(kept_rbcs)
            for i, cell in enumerate(kept_rbcs):
                max_same_iou = 0.0
                max_cross_iou = 0.0
                for j, other in enumerate(kept_rbcs):
                    if i == j:
                        continue
                    iou_val = compute_iou(cell["bbox"], other["bbox"])
                    if iou_val > 0.0:
                        if cell["class_id"] == other["class_id"]:
                            if iou_val > max_same_iou:
                                max_same_iou = iou_val
                        else:
                            if iou_val > max_cross_iou:
                                max_cross_iou = iou_val

                has_ambiguous = (
                    max_same_iou >= AMBIGUOUS_OVERLAP_IOU_THRESH
                    or max_cross_iou >= AMBIGUOUS_OVERLAP_IOU_THRESH
                )
                if max_same_iou >= AMBIGUOUS_OVERLAP_IOU_THRESH:
                    split_counts["ambiguous_same_class_overlaps_flagged"] += 1
                if max_cross_iou >= AMBIGUOUS_OVERLAP_IOU_THRESH:
                    split_counts["ambiguous_cross_class_overlaps_flagged"] += 1
                if has_ambiguous:
                    patient_summary[patient_id]["ambiguous_overlaps_flagged"] += 1
                    ambiguous_overlap_records.append({
                        "partition": split,
                        "source_image": fname,
                        "patient_id": patient_id,
                        "line_number": cell["line_number"],
                        "class_label": cell["class_label"],
                        "bbox": cell["bbox"],
                        "max_same_class_iou": round(max_same_iou, 4),
                        "max_cross_class_iou": round(max_cross_iou, 4)
                    })

                x1, y1, x2, y2 = cell["bbox"]
                px1 = max(0, x1 - PAD_PX)
                py1 = max(0, y1 - PAD_PX)
                px2 = min(img_w, x2 + PAD_PX)
                py2 = min(img_h, y2 + PAD_PX)

                raw_crop_np = img_np[py1:py2, px1:px2]
                masked_crop_np = apply_elliptical_mask(raw_crop_np)
                ch, cw = raw_crop_np.shape[:2]

                crop_id = f"{stem}_gt_{i + 1:04d}"
                cname = cell["class_label"]

                raw_rel = os.path.join("raw", split, cname, f"{crop_id}.png")
                masked_rel = os.path.join("masked", split, cname, f"{crop_id}.png")

                raw_abs = os.path.join(CROPS_DATASET_DIR, raw_rel)
                masked_abs = os.path.join(CROPS_DATASET_DIR, masked_rel)

                Image.fromarray(raw_crop_np).save(raw_abs, compress_level=1)
                Image.fromarray(masked_crop_np).save(masked_abs, compress_level=1)

                for mode_name, rel_p in [("raw", raw_rel), ("masked", masked_rel)]:
                    manifest_rows.append({
                        "crop_id": crop_id,
                        "source_image": fname,
                        "patient_id": patient_id,
                        "partition": split,
                        "class_id": cell["class_id"],
                        "class_label": cname,
                        "crop_mode": mode_name,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "padded_x1": px1,
                        "padded_y1": py1,
                        "padded_x2": px2,
                        "padded_y2": py2,
                        "crop_width": cw,
                        "crop_height": ch,
                        "max_same_class_iou": round(max_same_iou, 4),
                        "max_cross_class_iou": round(max_cross_iou, 4),
                        "has_ambiguous_overlap": has_ambiguous,
                        "crop_rel_path": rel_p
                    })

                split_counts["valid_rbc_crops_extracted"] += 1
                split_counts[cname] += 1
                patient_summary[patient_id][cname] += 1

            if img_idx % 25 == 0 or img_idx == len(img_files):
                print(
                    f"  [{split.upper()}] Processed {img_idx}/{len(img_files)} images | "
                    f"Valid RBCs so far: {split_counts['valid_rbc_crops_extracted']} "
                    f"(Uninfected: {split_counts['Uninfected']}, Parasitized: {split_counts['Parasitized']})"
                )

        split_counts["patients_count"] = len(split_patients_seen)
        split_counts["patients_list"] = sorted(list(split_patients_seen))
        split_summary[split] = split_counts

    # Write CSV manifest
    fieldnames = [
        "crop_id", "source_image", "patient_id", "partition",
        "class_id", "class_label", "crop_mode",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "padded_x1", "padded_y1", "padded_x2", "padded_y2",
        "crop_width", "crop_height",
        "max_same_class_iou", "max_cross_class_iou", "has_ambiguous_overlap",
        "crop_rel_path"
    ]
    with open(MANIFEST_CSV_PATH, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    elapsed = round(time.time() - t0, 2)

    report_data = {
        "phase": "Phase 6 — Domain-Matched Full-Smear RBC Crop Dataset Preparation",
        "source_yolo_dataset": YOLO_DIR,
        "output_crops_dataset_dir": CROPS_DATASET_DIR,
        "manifest_csv_path": MANIFEST_CSV_PATH,
        "parameters": {
            "pad_px": PAD_PX,
            "min_box_size_px": MIN_BOX_SIZE,
            "duplicate_suppression_iou_threshold": DUPLICATE_IOU_THRESH,
            "ambiguous_overlap_flag_iou_threshold": AMBIGUOUS_OVERLAP_IOU_THRESH,
            "crop_modes_saved": ["raw", "masked"],
            "class_mapping": {"0": "Uninfected", "1": "Parasitized"},
            "wbc_handling": "Excluded from binary malaria classification dataset (logged separately)"
        },
        "patient_leakage_verification": {
            "train_patients_count": split_summary["train"]["patients_count"],
            "val_patients_count": split_summary["val"]["patients_count"],
            "test_patients_count": split_summary["test"]["patients_count"],
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "verified_zero_leakage": True
        },
        "splits": split_summary,
        "per_patient_counts": dict(sorted(patient_summary.items())),
        "exclusions_and_overlaps_summary": {
            "total_excluded_wbcs": len(excluded_wbc_records),
            "total_invalid_boxes": len(invalid_box_records),
            "total_duplicate_annotations_suppressed": len(duplicate_suppressed_records),
            "total_ambiguous_overlaps_flagged": len(ambiguous_overlap_records),
            "duplicate_suppressed_details": duplicate_suppressed_records,
            "excluded_wbcs_sample": excluded_wbc_records[:15],
            "invalid_boxes_details": invalid_box_records
        },
        "total_manifest_rows_written": len(manifest_rows),
        "processing_time_seconds": elapsed
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(report_data, jf, indent=2)

    txt_lines = [
        "=" * 78,
        "SMARTCELL PHASE 6: DOMAIN-MATCHED RBC CROP DATASET PREPARATION REPORT",
        "=" * 78,
        f"Output Dataset Root:  {CROPS_DATASET_DIR}",
        f"CSV Manifest Path:    {MANIFEST_CSV_PATH}",
        f"JSON Report Path:     {REPORT_JSON_PATH}",
        f"Crop Padding:         {PAD_PX} px (clipped to 5312x2988 image bounds)",
        f"Crop Variants Saved:  1) raw (unmodified rectangular RGB)",
        f"                      2) masked (elliptical black-corner masked RGB)",
        f"Class Mapping:        0 = Uninfected, 1 = Parasitized (WBC class 2 excluded)",
        f"Patient Leakage:      0 overlapping patients across train/val/test (VERIFIED)",
        "",
        "------------------------------------------------------------------------------",
        "PARTITION SUMMARY (UNIQUE RBC CROPS PER VARIANT)",
        "------------------------------------------------------------------------------"
    ]
    for sp in ["train", "val", "test"]:
        sc = split_summary[sp]
        txt_lines.extend([
            f"  [{sp.upper()}]",
            f"    Patients / Images:           {sc['patients_count']} patients / {sc['images_count']} images",
            f"    Total Raw YOLO Annotations:  {sc['total_raw_annotations']}",
            f"    Valid RBC Crops Extracted:   {sc['valid_rbc_crops_extracted']} (saved in both 'raw' and 'masked')",
            f"      - Uninfected (0):          {sc['Uninfected']} ({sc['Uninfected'] / sc['valid_rbc_crops_extracted'] * 100:.2f}%)",
            f"      - Parasitized (1):         {sc['Parasitized']} ({sc['Parasitized'] / sc['valid_rbc_crops_extracted'] * 100:.2f}%)",
            f"    Excluded WBCs (Class 2):     {sc['excluded_wbc_count']}",
            f"    Duplicate Boxes Suppressed:  {sc['duplicate_boxes_suppressed']} (IoU >= {DUPLICATE_IOU_THRESH})",
            f"    Invalid / Tiny Boxes:        {sc['invalid_box_count']}",
            f"    Ambiguous Overlaps Flagged:  same-class={sc['ambiguous_same_class_overlaps_flagged']}, cross-class={sc['ambiguous_cross_class_overlaps_flagged']} (IoU >= {AMBIGUOUS_OVERLAP_IOU_THRESH})",
            ""
        ])
    txt_lines.extend([
        f"Total PNG Crop Files Saved:      {len(manifest_rows)} ({len(manifest_rows)//2} raw + {len(manifest_rows)//2} masked)",
        f"Total Processing Time:           {elapsed} s",
        "=" * 78
    ])
    txt_content = "\n".join(txt_lines) + "\n"
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as tf:
        tf.write(txt_content)

    print("\n" + txt_content)


if __name__ == "__main__":
    main()
