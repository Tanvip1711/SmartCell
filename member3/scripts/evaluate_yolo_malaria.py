"""
evaluate_yolo_malaria.py
------------------------
Comprehensive Evaluation, Prediction Visualization, and Report Generation
for SmartCell Phase 4 (YOLOv8n Object Detection).

Performs:
1. Validation set evaluation using best.pt.
2. Untouched Test set evaluation (25 images, 4,800 annotations, 5 isolated patients).
3. Side-by-side Ground Truth vs. Prediction visualizations (full-frame and 800x800 crops)
   highlighting true detections, missed cells (FN), false positives (FP), and dense regions.
4. Machine-readable JSON report and human-readable TXT report generation.
"""

import os
import sys
import csv
import json
import time
import platform
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

# Paths
BASE_DIR = r"D:\smartcell"
YOLO_DATA_DIR = r"D:\smartcell\yolo_malaria"
DATA_YAML = os.path.join(YOLO_DATA_DIR, "data.yaml")
RESULTS_DIR = r"D:\smartcell\results"
RUN_DIR = os.path.join(RESULTS_DIR, "yolo_malaria_run")
WEIGHTS_DIR = os.path.join(RUN_DIR, "weights")
BEST_PT = os.path.join(WEIGHTS_DIR, "best.pt")
LAST_PT = os.path.join(WEIGHTS_DIR, "last.pt")
PRED_DIR = os.path.join(RUN_DIR, "predictions")

REPORT_JSON_METRICS = os.path.join(RESULTS_DIR, "metrics", "yolo_malaria_training_report.json")
REPORT_JSON_ROOT = os.path.join(RESULTS_DIR, "yolo_malaria_training_report.json")
REPORT_TXT_ROOT = os.path.join(RESULTS_DIR, "yolo_malaria_training_report.txt")
REPORT_TXT_METRICS = os.path.join(RESULTS_DIR, "metrics", "yolo_malaria_training_report.txt")

PREP_REPORT_JSON = os.path.join(RESULTS_DIR, "metrics", "yolo_preparation_report.json")

CLASS_NAMES = {
    0: "Uninfected RBC",
    1: "Parasitized RBC",
    2: "WBC"
}

CLASS_COLORS = {
    0: (46, 204, 113),   # Green (BGR)
    1: (46, 46, 231),    # Red/Crimson (BGR)
    2: (230, 126, 34)    # Blue/Orange (BGR)
}

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DET = 350


def load_gt_boxes(lbl_path, img_w, img_h):
    """Loads ground truth YOLO boxes in pixel coordinates [cls_id, x1, y1, x2, y2]."""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc = float(parts[1]) * img_w
            yc = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
            x1 = int(round(xc - bw / 2.0))
            y1 = int(round(yc - bh / 2.0))
            x2 = int(round(xc + bw / 2.0))
            y2 = int(round(yc + bh / 2.0))
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def extract_metrics_dict(val_results, class_instance_counts):
    """Extracts overall and per-class metrics from Ultralytics DetMetrics object."""
    box = val_results.box
    overall = {
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "mAP50": round(float(box.map50), 6),
        "mAP50_95": round(float(box.map), 6),
    }

    per_class = {}
    # box.ap_class_index contains indices of classes present in the evaluation split
    ap_cls_indices = [int(x) for x in box.ap_class_index]
    for idx_pos, cls_id in enumerate(ap_cls_indices):
        p, r, ap50, ap = box.class_result(idx_pos)
        cname = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        per_class[cname] = {
            "class_id": cls_id,
            "instances": class_instance_counts.get(cname, 0),
            "precision": round(float(p), 6),
            "recall": round(float(r), 6),
            "mAP50": round(float(ap50), 6),
            "mAP50_95": round(float(ap), 6)
        }

    speed = {k: round(float(v), 3) for k, v in val_results.speed.items()}
    return overall, per_class, speed


def generate_prediction_visualizations(model):
    r"""
    Generates full-frame and high-res 800x800 crop comparisons on test images.
    Saves outputs to D:\smartcell\results\yolo_malaria_run\predictions.
    """
    os.makedirs(PRED_DIR, exist_ok=True)
    test_img_dir = os.path.join(YOLO_DATA_DIR, "images", "test")
    test_lbl_dir = os.path.join(YOLO_DATA_DIR, "labels", "test")

    test_imgs = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")])

    # Select diverse test images: high parasitemia, low parasitemia, WBC presence, healthy control
    selected = []
    for img_f in test_imgs:
        stem = os.path.splitext(img_f)[0]
        lbl_f = os.path.join(test_lbl_dir, f"{stem}.txt")
        with open(lbl_f, "r") as f:
            classes = [int(l.split()[0]) for l in f if l.strip()]
        n_par = sum(1 for c in classes if c == 1)
        n_wbc = sum(1 for c in classes if c == 2)
        selected.append((img_f, stem, len(classes), n_par, n_wbc))

    # Sort to get images with high parasitized, images with WBC, and control images
    selected.sort(key=lambda x: (x[4] > 0, x[3]), reverse=True)
    sample_list = selected[:4] + selected[-2:]  # Top 4 infected/WBC + 2 low/uninfected

    saved_files = []

    for img_f, stem, total_gt, n_par_gt, n_wbc_gt in sample_list:
        img_path = os.path.join(test_img_dir, img_f)
        lbl_path = os.path.join(test_lbl_dir, f"{stem}.txt")

        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]

        gt_boxes = load_gt_boxes(lbl_path, w, h)

        # Run inference
        preds = model.predict(
            source=img_path,
            imgsz=640,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            verbose=False
        )[0]

        pred_boxes = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            xyxy = preds.boxes.xyxy.cpu().numpy()
            confs = preds.boxes.conf.cpu().numpy()
            clss = preds.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
                pred_boxes.append((int(cl), int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), float(cf)))

        # Draw Ground Truth canvas
        gt_canvas = orig_img.copy()
        for cl, x1, y1, x2, y2 in gt_boxes:
            thick = 2 if cl == 0 else 5
            cv2.rectangle(gt_canvas, (x1, y1), (x2, y2), CLASS_COLORS[cl], thick)
            if cl in (1, 2):
                cv2.putText(gt_canvas, f"GT: {CLASS_NAMES[cl]}", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLASS_COLORS[cl], 3, cv2.LINE_AA)

        # Draw Prediction canvas
        pred_canvas = orig_img.copy()
        pred_counts = {0: 0, 1: 0, 2: 0}
        for cl, x1, y1, x2, y2, cf in pred_boxes:
            pred_counts[cl] = pred_counts.get(cl, 0) + 1
            thick = 2 if cl == 0 else 5
            cv2.rectangle(pred_canvas, (x1, y1), (x2, y2), CLASS_COLORS[cl], thick)
            if cl in (1, 2):
                cv2.putText(pred_canvas, f"{CLASS_NAMES[cl]} {cf:.2f}", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLASS_COLORS[cl], 3, cv2.LINE_AA)

        # Add header banners to both canvases
        cv2.rectangle(gt_canvas, (20, 20), (1400, 130), (20, 20, 20), -1)
        cv2.putText(gt_canvas, f"GROUND TRUTH ({stem}) | Total: {total_gt} | Par: {n_par_gt} | WBC: {n_wbc_gt}",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.rectangle(pred_canvas, (20, 20), (1650, 130), (20, 20, 20), -1)
        cv2.putText(pred_canvas,
                    f"YOLOv8n PREDICTION (conf>={CONF_THRESHOLD}) | Total: {len(pred_boxes)} | Par: {pred_counts[1]} | WBC: {pred_counts[2]}",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 255), 3, cv2.LINE_AA)

        # Resize and stack side-by-side (1920x540 total)
        disp_w, disp_h = 1280, int(1280 * h / w)
        gt_small = cv2.resize(gt_canvas, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        pred_small = cv2.resize(pred_canvas, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        comparison = np.hstack([gt_small, pred_small])

        comp_path = os.path.join(PRED_DIR, f"comparison_full_{stem}.jpg")
        cv2.imwrite(comp_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved_files.append(comp_path)

        # Also save standalone predicted full-frame overview
        pred_full_path = os.path.join(PRED_DIR, f"pred_overview_{stem}.jpg")
        cv2.imwrite(pred_full_path, pred_small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved_files.append(pred_full_path)

        # High-resolution 800x800 zoomed side-by-side crops around Parasitized / WBC / dense regions
        interesting_gt = [b for b in gt_boxes if b[0] in (1, 2)]
        if not interesting_gt and gt_boxes:
            interesting_gt = [gt_boxes[len(gt_boxes) // 2]]

        for c_idx, (cl, x1, y1, x2, y2) in enumerate(interesting_gt[:2], 1):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop_sz = 800
            cx1 = max(0, min(w - crop_sz, cx - crop_sz // 2))
            cy1 = max(0, min(h - crop_sz, cy - crop_sz // 2))
            cx2, cy2 = cx1 + crop_sz, cy1 + crop_sz

            crop_gt = orig_img[cy1:cy2, cx1:cx2].copy()
            crop_pr = orig_img[cy1:cy2, cx1:cx2].copy()

            # Draw GT boxes in crop
            for g_cl, gx1, gy1, gx2, gy2 in gt_boxes:
                if not (gx2 < cx1 or gx1 > cx2 or gy2 < cy1 or gy1 > cy2):
                    rx1, ry1 = max(0, gx1 - cx1), max(0, gy1 - cy1)
                    rx2, ry2 = min(crop_sz, gx2 - cx1), min(crop_sz, gy2 - cy1)
                    cv2.rectangle(crop_gt, (rx1, ry1), (rx2, ry2), CLASS_COLORS[g_cl], 3 if g_cl else 2)
                    if g_cl in (1, 2):
                        cv2.putText(crop_gt, f"GT:{CLASS_NAMES[g_cl]}", (rx1, max(20, ry1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS[g_cl], 2, cv2.LINE_AA)

            # Draw Pred boxes in crop with confidence scores
            for p_cl, px1, py1, px2, py2, p_cf in pred_boxes:
                if not (px2 < cx1 or px1 > cx2 or py2 < cy1 or py1 > cy2):
                    rx1, ry1 = max(0, px1 - cx1), max(0, py1 - cy1)
                    rx2, ry2 = min(crop_sz, px2 - cx1), min(crop_sz, py2 - cy1)
                    cv2.rectangle(crop_pr, (rx1, ry1), (rx2, ry2), CLASS_COLORS[p_cl], 3 if p_cl else 2)
                    lbl_txt = f"{CLASS_NAMES[p_cl][:3]} {p_cf:.2f}" if p_cl == 0 else f"{CLASS_NAMES[p_cl]} {p_cf:.2f}"
                    cv2.putText(crop_pr, lbl_txt, (rx1, max(20, ry1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55 if p_cl == 0 else 0.65,
                                CLASS_COLORS[p_cl], 2, cv2.LINE_AA)

            cv2.rectangle(crop_gt, (0, 0), (crop_sz, 36), (20, 20, 20), -1)
            cv2.putText(crop_gt, f"Ground Truth Crop ({stem})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(crop_pr, (0, 0), (crop_sz, 36), (20, 20, 20), -1)
            cv2.putText(crop_pr, f"YOLOv8n Prediction Crop (conf>={CONF_THRESHOLD})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2, cv2.LINE_AA)

            crop_pair = np.hstack([crop_gt, crop_pr])
            crop_path = os.path.join(PRED_DIR, f"crop_compare_{stem}_roi{c_idx}.jpg")
            cv2.imwrite(crop_path, crop_pair, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_files.append(crop_path)

    return saved_files


def parse_training_history(csv_path):
    """Reads Ultralytics results.csv to find best epoch and loss progression."""
    if not os.path.exists(csv_path):
        return {}
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clean_r = {k.strip(): v.strip() for k, v in r.items()}
            rows.append(clean_r)
    if not rows:
        return {}

    # Find best epoch by fitness (0.1 * mAP50 + 0.9 * mAP50-95)
    best_row = max(
        rows,
        key=lambda r: 0.1 * float(r.get("metrics/mAP50(B)", 0)) + 0.9 * float(r.get("metrics/mAP50-95(B)", 0))
    )
    first_row = rows[0]
    last_row = rows[-1]

    return {
        "epochs_completed": len(rows),
        "best_epoch": int(float(best_row.get("epoch", 0))),
        "first_epoch_losses": {
            "train_box_loss": float(first_row.get("train/box_loss", 0)),
            "train_cls_loss": float(first_row.get("train/cls_loss", 0)),
            "train_dfl_loss": float(first_row.get("train/dfl_loss", 0)),
            "val_box_loss": float(first_row.get("val/box_loss", 0)),
            "val_cls_loss": float(first_row.get("val/cls_loss", 0)),
            "val_dfl_loss": float(first_row.get("val/dfl_loss", 0)),
        },
        "best_epoch_metrics": {
            "epoch": int(float(best_row.get("epoch", 0))),
            "train_box_loss": float(best_row.get("train/box_loss", 0)),
            "train_cls_loss": float(best_row.get("train/cls_loss", 0)),
            "train_dfl_loss": float(best_row.get("train/dfl_loss", 0)),
            "val_box_loss": float(best_row.get("val/box_loss", 0)),
            "val_cls_loss": float(best_row.get("val/cls_loss", 0)),
            "val_dfl_loss": float(best_row.get("val/dfl_loss", 0)),
            "precision": float(best_row.get("metrics/precision(B)", 0)),
            "recall": float(best_row.get("metrics/recall(B)", 0)),
            "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(best_row.get("metrics/mAP50-95(B)", 0)),
        },
        "final_epoch_losses": {
            "epoch": int(float(last_row.get("epoch", 0))),
            "train_box_loss": float(last_row.get("train/box_loss", 0)),
            "train_cls_loss": float(last_row.get("train/cls_loss", 0)),
            "train_dfl_loss": float(last_row.get("train/dfl_loss", 0)),
            "val_box_loss": float(last_row.get("val/box_loss", 0)),
            "val_cls_loss": float(last_row.get("val/cls_loss", 0)),
            "val_dfl_loss": float(last_row.get("val/dfl_loss", 0)),
        }
    }


def main():
    print("=" * 70)
    print("SMARTCELL PHASE 4: YOLOv8n EVALUATION & REPORT GENERATION")
    print("=" * 70)

    if not os.path.exists(BEST_PT):
        raise FileNotFoundError(f"Trained checkpoint not found at: {BEST_PT}")

    with open(PREP_REPORT_JSON, "r", encoding="utf-8") as f:
        prep_report = json.load(f)

    timing_path = os.path.join(RUN_DIR, "training_timing.json")
    timing_info = {}
    if os.path.exists(timing_path):
        with open(timing_path, "r", encoding="utf-8") as f:
            timing_info = json.load(f)

    history_summary = parse_training_history(os.path.join(RUN_DIR, "results.csv"))

    model = YOLO(BEST_PT)

    # 1. Validation Set Evaluation
    print("\n[1/3] Evaluating best.pt on Validation Set (25 images, 5,310 annotations)...")
    val_res = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640,
        batch=4,
        rect=True,
        max_det=MAX_DET,
        device="cpu",
        project=RUN_DIR,
        name="val_eval",
        exist_ok=True,
        plots=True
    )
    val_cls_counts = prep_report["splits"]["val"]["annotations_by_class"]
    val_overall, val_per_class, val_speed = extract_metrics_dict(val_res, val_cls_counts)

    # 2. Untouched Test Set Evaluation
    print("\n[2/3] Evaluating best.pt on Untouched Test Set (25 images, 4,800 annotations)...")
    test_res = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=4,
        rect=True,
        max_det=MAX_DET,
        device="cpu",
        project=RUN_DIR,
        name="test_eval",
        exist_ok=True,
        plots=True
    )
    test_cls_counts = prep_report["splits"]["test"]["annotations_by_class"]
    test_overall, test_per_class, test_speed = extract_metrics_dict(test_res, test_cls_counts)

    # 3. Prediction Visualizations
    print("\n[3/3] Generating prediction visualizations on Test Set...")
    pred_files = generate_prediction_visualizations(model)
    print(f"Generated {len(pred_files)} visualization files in {PRED_DIR}")

    # Build comprehensive report
    best_size_bytes = os.path.getsize(BEST_PT) if os.path.exists(BEST_PT) else None
    last_size_bytes = os.path.getsize(LAST_PT) if os.path.exists(LAST_PT) else None

    report = {
        "phase": "Phase 4: YOLOv8n Baseline Object Detection",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment_configuration": {
            "task": "detect",
            "model_architecture": "YOLOv8n (yolov8n.pt)",
            "parameters": 3006233,
            "layers_fused": 72,
            "gflops": 8.1,
            "dataset_yaml": DATA_YAML,
            "epochs": 50,
            "imgsz": 640,
            "effective_rectangular_size": "640x384",
            "batch_size": 4,
            "rect": True,
            "max_det": MAX_DET,
            "optimizer": "AdamW (auto: lr=0.001429, momentum=0.9)",
            "seed": 42,
            "workers": 0,
            "cache": "ram",
            "prediction_confidence_threshold": CONF_THRESHOLD,
            "prediction_iou_threshold": IOU_THRESHOLD
        },
        "software_and_hardware": {
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "ultralytics_version": ultralytics.__version__,
            "opencv_version": cv2.__version__,
            "cuda_available": torch.cuda.is_available(),
            "training_device": "CPU (12th Gen Intel Core i5-12450H)",
            "gpu_hardware_present": "NVIDIA GeForce RTX 2050 Laptop GPU (4096 MiB) — unused due to CPU-only PyTorch build in environment"
        },
        "dataset_summary": {
            "source": "NIH-NLM Thin Blood Smears Pf (Polygon Set)",
            "original_image_resolution": "5312x2988",
            "total_patients": 33,
            "total_images": 165,
            "total_annotations": 34264,
            "patient_leakage_overlap": 0,
            "class_mapping": CLASS_NAMES,
            "splits": prep_report["splits"]
        },
        "training_summary": {
            "training_duration_seconds": timing_info.get("training_duration_seconds"),
            "training_duration_minutes": timing_info.get("training_duration_minutes"),
            "training_duration_hours": timing_info.get("training_duration_hours"),
            "epochs_completed": history_summary.get("epochs_completed"),
            "best_epoch": history_summary.get("best_epoch"),
            "first_epoch_losses": history_summary.get("first_epoch_losses"),
            "best_epoch_metrics": history_summary.get("best_epoch_metrics"),
            "final_epoch_losses": history_summary.get("final_epoch_losses"),
            "best_checkpoint_size_bytes": best_size_bytes,
            "best_checkpoint_size_mb": round(best_size_bytes / (1024 * 1024), 2) if best_size_bytes else None,
            "last_checkpoint_size_bytes": last_size_bytes
        },
        "validation_evaluation": {
            "split": "val",
            "images_evaluated": 25,
            "patients_evaluated": 5,
            "total_annotations": 5310,
            "overall_metrics": val_overall,
            "per_class_metrics": val_per_class,
            "inference_speed_ms_per_image": val_speed
        },
        "test_evaluation": {
            "split": "test",
            "images_evaluated": 25,
            "patients_evaluated": 5,
            "total_annotations": 4800,
            "overall_metrics": test_overall,
            "per_class_metrics": test_per_class,
            "inference_speed_ms_per_image": test_speed
        },
        "generated_artifacts": {
            "experiment_directory": RUN_DIR,
            "smoke_test_directory": os.path.join(RESULTS_DIR, "yolo_smoke_test"),
            "best_checkpoint": BEST_PT,
            "last_checkpoint": LAST_PT,
            "training_results_csv": os.path.join(RUN_DIR, "results.csv"),
            "training_curves_plot": os.path.join(RUN_DIR, "results.png"),
            "val_evaluation_directory": os.path.join(RUN_DIR, "val_eval"),
            "val_confusion_matrix": os.path.join(RUN_DIR, "val_eval", "confusion_matrix.png"),
            "val_confusion_matrix_normalized": os.path.join(RUN_DIR, "val_eval", "confusion_matrix_normalized.png"),
            "test_evaluation_directory": os.path.join(RUN_DIR, "test_eval"),
            "test_confusion_matrix": os.path.join(RUN_DIR, "test_eval", "confusion_matrix.png"),
            "test_confusion_matrix_normalized": os.path.join(RUN_DIR, "test_eval", "confusion_matrix_normalized.png"),
            "prediction_visualizations_directory": PRED_DIR,
            "prediction_visualization_files": pred_files
        },
        "warnings_and_resolutions": [
            "Max detections warning: Dataset images contain up to 313 cells (exceeding Ultralytics default max_det=300). Resolved by explicitly setting max_det=350 during training, validation, testing, and prediction.",
            "Environment isolation: Created D:\\smartcell\\venv_yolo (Python 3.13.3 with torch 2.14.0+cpu and ultralytics 8.4.160) to prevent any package conflict with D:\\smartcell\\venv (Python 3.11.9 with TensorFlow 2.21.0).",
            "Aspect ratio optimization: Enabled rectangular training (rect=True, 640x384) to match the native 16:9 (5312x2988) aspect ratio, eliminating 40% letterbox padding overhead and preventing 2x2 mosaic downscaling of tiny parasites."
        ],
        "known_limitations": [
            "Severe Class Imbalance: Uninfected RBCs account for 96.52% (33,071) of annotations, whereas Parasitized RBCs account for 3.33% (1,142) and WBCs only 0.15% (51 total; 37 train, 8 val, 6 test).",
            "Resolution Downscaling: Downsampling 5312x2988 microscopic fields to 640x384 reduces average RBC diameter from ~130px to ~15px, making subtle intracellular Plasmodium ring inclusions difficult to distinguish from uninfected RBCs at the detection stage alone.",
            "Limited Test Partition Size: The test set contains 25 images from 5 patients (with only 6 WBC instances and 376 Parasitized RBC instances), introducing statistical variance in per-class mAP estimates.",
            "Two-Stage Pipeline Necessity: High Uninfected RBC localization recall confirms YOLOv8n works effectively as a Stage-1 cell localizer and counter, while confirming why SmartCell uses the Stage-2 Baseline CNN (95.04% accuracy at 224x224 per cell crop) for definitive infection classification."
        ]
    }

    os.makedirs(os.path.dirname(REPORT_JSON_METRICS), exist_ok=True)
    with open(REPORT_JSON_METRICS, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(REPORT_JSON_ROOT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write human-readable TXT report
    txt_lines = [
        "=" * 75,
        "SMARTCELL PHASE 4: YOLOv8n OBJECT DETECTION TRAINING & EVALUATION REPORT",
        "=" * 75,
        f"Timestamp:          {report['timestamp']}",
        f"Model:              YOLOv8n (yolov8n.pt) — 3,006,233 parameters (8.1 GFLOPs)",
        f"Environment:        {sys.executable} (Python {platform.python_version()})",
        f"Libraries:          PyTorch {torch.__version__} | Ultralytics {ultralytics.__version__}",
        f"Hardware Device:    CPU (12th Gen Intel Core i5-12450H)",
        f"Dataset Config:     {DATA_YAML}",
        f"Training Setup:     50 epochs | imgsz=640 (rect 640x384) | batch=4 | max_det=350 | seed=42",
        f"Training Duration:  {report['training_summary']['training_duration_seconds']}s "
        f"({report['training_summary']['training_duration_minutes']} min)",
        f"Best Epoch:         Epoch {report['training_summary']['best_epoch']} / {report['training_summary']['epochs_completed']}",
        f"Best Checkpoint:    {BEST_PT} ({report['training_summary']['best_checkpoint_size_mb']} MB)",
        "",
        "-" * 75,
        "1. VALIDATION SET METRICS (25 Images, 5 Patients, 5,310 Annotations)",
        "-" * 75,
        f"  Overall Precision:  {val_overall['precision']:.4f} ({val_overall['precision']*100:.2f}%)",
        f"  Overall Recall:     {val_overall['recall']:.4f} ({val_overall['recall']*100:.2f}%)",
        f"  Overall mAP@50:     {val_overall['mAP50']:.4f} ({val_overall['mAP50']*100:.2f}%)",
        f"  Overall mAP@50-95:  {val_overall['mAP50_95']:.4f} ({val_overall['mAP50_95']*100:.2f}%)",
        "",
        "  Per-Class Validation Breakdown:",
        f"  {'Class Name':<18} | {'Instances':>9} | {'Precision':>9} | {'Recall':>9} | {'mAP@50':>9} | {'mAP@50-95':>9}",
        "  " + "-" * 71
    ]
    for cname in ["Uninfected RBC", "Parasitized RBC", "WBC"]:
        cinfo = val_per_class.get(cname, {})
        txt_lines.append(
            f"  {cname:<18} | {cinfo.get('instances', 0):>9} | "
            f"{cinfo.get('precision', 0):>9.4f} | {cinfo.get('recall', 0):>9.4f} | "
            f"{cinfo.get('mAP50', 0):>9.4f} | {cinfo.get('mAP50_95', 0):>9.4f}"
        )

    txt_lines.extend([
        "",
        "-" * 75,
        "2. UNTOUCHED TEST SET METRICS (25 Images, 5 Patients, 4,800 Annotations)",
        "-" * 75,
        f"  Overall Precision:  {test_overall['precision']:.4f} ({test_overall['precision']*100:.2f}%)",
        f"  Overall Recall:     {test_overall['recall']:.4f} ({test_overall['recall']*100:.2f}%)",
        f"  Overall mAP@50:     {test_overall['mAP50']:.4f} ({test_overall['mAP50']*100:.2f}%)",
        f"  Overall mAP@50-95:  {test_overall['mAP50_95']:.4f} ({test_overall['mAP50_95']*100:.2f}%)",
        f"  Inference Speed:    {test_speed.get('preprocess', 0):.1f}ms preprocess | "
        f"{test_speed.get('inference', 0):.1f}ms inference | {test_speed.get('postprocess', 0):.1f}ms postprocess",
        "",
        "  Per-Class Test Breakdown:",
        f"  {'Class Name':<18} | {'Instances':>9} | {'Precision':>9} | {'Recall':>9} | {'mAP@50':>9} | {'mAP@50-95':>9}",
        "  " + "-" * 71
    ])
    for cname in ["Uninfected RBC", "Parasitized RBC", "WBC"]:
        cinfo = test_per_class.get(cname, {})
        txt_lines.append(
            f"  {cname:<18} | {cinfo.get('instances', 0):>9} | "
            f"{cinfo.get('precision', 0):>9.4f} | {cinfo.get('recall', 0):>9.4f} | "
            f"{cinfo.get('mAP50', 0):>9.4f} | {cinfo.get('mAP50_95', 0):>9.4f}"
        )

    txt_lines.extend([
        "",
        "-" * 75,
        "3. GENERATED ARTIFACTS & OUTPUT PATHS",
        "-" * 75,
        f"  Best Model Weights:       {BEST_PT}",
        f"  Last Model Weights:       {LAST_PT}",
        f"  Training History CSV:     {os.path.join(RUN_DIR, 'results.csv')}",
        f"  Training Curves Plot:     {os.path.join(RUN_DIR, 'results.png')}",
        f"  Val Evaluation Dir:       {os.path.join(RUN_DIR, 'val_eval')}",
        f"  Test Evaluation Dir:      {os.path.join(RUN_DIR, 'test_eval')}",
        f"  Prediction Visuals Dir:   {PRED_DIR} ({len(pred_files)} images)",
        f"  JSON Report (Metrics):    {REPORT_JSON_METRICS}",
        f"  TXT Report:               {REPORT_TXT_ROOT}",
        "=" * 75
    ])

    txt_content = "\n".join(txt_lines) + "\n"
    with open(REPORT_TXT_ROOT, "w", encoding="utf-8") as f:
        f.write(txt_content)
    with open(REPORT_TXT_METRICS, "w", encoding="utf-8") as f:
        f.write(txt_content)

    print("\n" + txt_content)


if __name__ == "__main__":
    main()
