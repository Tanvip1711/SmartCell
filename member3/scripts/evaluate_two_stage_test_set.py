r"""
evaluate_two_stage_test_set.py
------------------------------
SmartCell Phase 6: End-to-End Two-Stage Pipeline Evaluation on All 25 Held-Out Test Images
(Stage 1 YOLOv8n on NVIDIA GeForce RTX 2050 GPU [venv_yolo_gpu] + Stage 2 CNN in [venv] CPU)

Evaluates end-to-end performance separately from ground-truth crop classification,
explicitly accounting for:
1. Stage 1 YOLO detection misses (unmatched ground-truth Uninfected and Parasitized RBCs).
2. Stage 1 spurious / unmatched detections.
3. End-to-end Parasitized sensitivity, Uninfected specificity, Precision, and F1 across
   Strategies A, B, C, and D when Stage 1 misses are penalized as False Negatives.
4. Image-level (25 test images) and Patient-level (5 held-out test patients) parasitemia
   estimation accuracy (Mean Absolute Error vs. ground-truth parasitemia).
"""

import os
import sys
import csv
import json
import time
import argparse
import subprocess
from collections import defaultdict
import numpy as np
from PIL import Image

BASE_DIR = r"D:\smartcell"
VENV_YOLO_GPU_PYTHON = os.path.join(BASE_DIR, "venv_yolo_gpu", "Scripts", "python.exe")
VENV_TF_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")

POLYGON_SET_DIR = os.path.join(BASE_DIR, "Polygon Set")
YOLO_TEST_IMG_DIR = os.path.join(BASE_DIR, "yolo_malaria", "images", "test")
YOLO_TEST_LBL_DIR = os.path.join(BASE_DIR, "yolo_malaria", "labels", "test")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "results", "yolo_malaria_run", "weights", "best.pt")

ORIG_CNN_PATH = os.path.join(BASE_DIR, "models", "malaria_cnn_best.keras")
FINETUNED_RAW_PATH = os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_raw_best.keras")
FINETUNED_MASKED_PATH = os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_masked_best.keras")

OUT_DIR = os.path.join(BASE_DIR, "results", "domain_adaptation", "two_stage_test_eval")
STAGE1_JSON = os.path.join(OUT_DIR, "stage1_gpu_test_detections.json")
STAGE1_CROPS_NPZ = os.path.join(OUT_DIR, "stage1_gpu_test_crops.npz")
REPORT_JSON_PATH = os.path.join(BASE_DIR, "results", "metrics", "phase6_two_stage_end_to_end_report.json")
REPORT_TXT_PATH = os.path.join(BASE_DIR, "results", "domain_adaptation", "phase6_two_stage_end_to_end_report.txt")
PER_IMAGE_CSV_PATH = os.path.join(OUT_DIR, "two_stage_test_25images_comparison.csv")

PAD_PX = 4
MIN_BOX_SIZE = 16
YOLO_CONF = 0.25
YOLO_IOU = 0.45
DEDUP_IOU = 0.50
MATCH_GT_IOU = 0.40
MAX_DET = 350


def compute_iou(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = float((x_right - x_left) * (y_bottom - y_top))
    area_a = float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def apply_elliptical_mask(crop_rgb_np):
    ch, cw = crop_rgb_np.shape[:2]
    yy, xx = np.ogrid[:ch, :cw]
    cy, cx = ch / 2.0, cw / 2.0
    ry, rx = ch / 2.0, cw / 2.0
    mask = ((xx - cx) ** 2 / (rx ** 2 + 1e-6) + (yy - cy) ** 2 / (ry ** 2 + 1e-6)) <= 1.0
    masked = crop_rgb_np.copy()
    masked[~mask] = 0
    return masked


def build_image_to_patient_map():
    img_to_patient = {}
    for p in sorted(os.listdir(POLYGON_SET_DIR)):
        img_dir = os.path.join(POLYGON_SET_DIR, p, "Img")
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_to_patient[os.path.splitext(fname)[0]] = p
                img_to_patient[fname] = p
    return img_to_patient


def run_stage1_gpu():
    """Runs in venv_yolo_gpu using NVIDIA GeForce RTX 2050 (device=0, conservative batch=1)."""
    import torch
    from ultralytics import YOLO

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    cuda_avail = torch.cuda.is_available()
    if not cuda_avail:
        raise RuntimeError("CUDA is not available in venv_yolo_gpu!")

    device_name = torch.cuda.get_device_name(0)
    free_before, total_vram = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats(0)

    print("=" * 80, flush=True)
    print(f"STAGE 1 GPU EVALUATION ON 25 HELD-OUT TEST IMAGES ({device_name})", flush=True)
    print(f"PyTorch: {torch.__version__} | CUDA Available: {cuda_avail} | Device: 0", flush=True)
    print(f"VRAM Before Load: {free_before / (1024**2):.1f} MB free / {total_vram / (1024**2):.1f} MB total", flush=True)
    print("=" * 80, flush=True)

    img_to_patient = build_image_to_patient_map()
    model = YOLO(YOLO_WEIGHTS)

    test_files = sorted([
        f for f in os.listdir(YOLO_TEST_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    all_raw_crops_224 = []
    all_masked_crops_224 = []
    per_image_records = []
    global_crop_idx = 0

    total_gt_uninf = 0
    total_gt_par = 0
    total_gt_wbc = 0
    total_matched_uninf = 0
    total_matched_par = 0
    total_missed_uninf = 0
    total_missed_par = 0
    total_spurious_dets = 0

    for idx, fname in enumerate(test_files, start=1):
        stem = os.path.splitext(fname)[0]
        patient_id = img_to_patient[stem]
        img_path = os.path.join(YOLO_TEST_IMG_DIR, fname)
        lbl_path = os.path.join(YOLO_TEST_LBL_DIR, f"{stem}.txt")

        with Image.open(img_path) as pil_im:
            rgb_im = pil_im.convert("RGB")
            img_w, img_h = rgb_im.size
            orig_np = np.array(rgb_im)

        # Parse ground truth
        gt_rbcs = []
        gt_wbcs = 0
        with open(lbl_path, "r", encoding="utf-8") as lf:
            for line_idx, line in enumerate(lf):
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                xc, yc, nw, nh = map(float, parts[1:5])
                bw_f, bh_f = nw * img_w, nh * img_h
                x1 = max(0, min(img_w, int(round(xc * img_w - bw_f / 2.0))))
                y1 = max(0, min(img_h, int(round(yc * img_h - bh_f / 2.0))))
                x2 = max(0, min(img_w, int(round(xc * img_w + bw_f / 2.0))))
                y2 = max(0, min(img_h, int(round(yc * img_h + bh_f / 2.0))))
                if cid == 2:
                    gt_wbcs += 1
                elif cid in (0, 1):
                    gt_rbcs.append({
                        "gt_idx": len(gt_rbcs),
                        "class_id": cid,
                        "bbox": [x1, y1, x2, y2]
                    })

        # Conservative single-image GPU inference (batch=1) for 4 GB RTX 2050
        t_img0 = time.time()
        res = model.predict(
            source=img_path,
            imgsz=640,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            max_det=MAX_DET,
            rect=True,
            device=0,
            verbose=False
        )[0]
        infer_ms = round((time.time() - t_img0) * 1000.0, 2)

        raw_dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
                cx1 = max(0, min(img_w, int(round(float(x1)))))
                cy1 = max(0, min(img_h, int(round(float(y1)))))
                cx2 = max(0, min(img_w, int(round(float(x2)))))
                cy2 = max(0, min(img_h, int(round(float(y2)))))
                if (cx2 - cx1) < MIN_BOX_SIZE or (cy2 - cy1) < MIN_BOX_SIZE:
                    continue
                if int(cl) == 2:
                    continue
                if int(cl) in (0, 1):
                    raw_dets.append({
                        "yolo_class_id": int(cl),
                        "yolo_conf": float(cf),
                        "bbox": [cx1, cy1, cx2, cy2]
                    })

        # Deduplicate candidate RBCs
        raw_dets.sort(key=lambda d: d["yolo_conf"], reverse=True)
        dedup_dets = []
        for d in raw_dets:
            if not any(compute_iou(d["bbox"], k["bbox"]) >= DEDUP_IOU for k in dedup_dets):
                dedup_dets.append(d)

        # Match deduplicated detections to ground-truth RBCs (greedy IoU >= 0.40)
        matched_gt_indices = set()
        det_records = []
        for d in dedup_dets:
            best_iou = 0.0
            best_gt = None
            for g in gt_rbcs:
                if g["gt_idx"] in matched_gt_indices:
                    continue
                iou_val = compute_iou(d["bbox"], g["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = g
            if best_gt is not None and best_iou >= MATCH_GT_IOU:
                matched_gt_indices.add(best_gt["gt_idx"])
                matched_gt_cls = best_gt["class_id"]
                match_iou = round(best_iou, 4)
            else:
                matched_gt_cls = -1  # Unmatched / spurious detection
                match_iou = round(best_iou, 4)

            x1, y1, x2, y2 = d["bbox"]
            px1 = max(0, x1 - PAD_PX)
            py1 = max(0, y1 - PAD_PX)
            px2 = min(img_w, x2 + PAD_PX)
            py2 = min(img_h, y2 + PAD_PX)

            raw_crop = orig_np[py1:py2, px1:px2]
            masked_crop = apply_elliptical_mask(raw_crop)

            raw_224 = np.array(Image.fromarray(raw_crop).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)
            masked_224 = np.array(Image.fromarray(masked_crop).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)

            all_raw_crops_224.append(raw_224)
            all_masked_crops_224.append(masked_224)

            det_records.append({
                "global_crop_idx": global_crop_idx,
                "yolo_class_id": d["yolo_class_id"],
                "yolo_conf": round(d["yolo_conf"], 6),
                "bbox": d["bbox"],
                "matched_gt_class_id": matched_gt_cls,
                "match_iou": match_iou
            })
            global_crop_idx += 1

        img_gt_uninf = sum(1 for g in gt_rbcs if g["class_id"] == 0)
        img_gt_par = sum(1 for g in gt_rbcs if g["class_id"] == 1)
        img_matched_uninf = sum(1 for d in det_records if d["matched_gt_class_id"] == 0)
        img_matched_par = sum(1 for d in det_records if d["matched_gt_class_id"] == 1)
        img_missed_uninf = img_gt_uninf - img_matched_uninf
        img_missed_par = img_gt_par - img_matched_par
        img_spurious = sum(1 for d in det_records if d["matched_gt_class_id"] == -1)

        total_gt_uninf += img_gt_uninf
        total_gt_par += img_gt_par
        total_gt_wbc += gt_wbcs
        total_matched_uninf += img_matched_uninf
        total_matched_par += img_matched_par
        total_missed_uninf += img_missed_uninf
        total_missed_par += img_missed_par
        total_spurious_dets += img_spurious

        per_image_records.append({
            "image_filename": fname,
            "stem": stem,
            "patient_id": patient_id,
            "gpu_inference_ms": infer_ms,
            "gt_uninfected_rbc": img_gt_uninf,
            "gt_parasitized_rbc": img_gt_par,
            "gt_total_rbc": img_gt_uninf + img_gt_par,
            "gt_wbc": gt_wbcs,
            "yolo_valid_rbc_candidates": len(det_records),
            "matched_gt_uninfected": img_matched_uninf,
            "matched_gt_parasitized": img_matched_par,
            "missed_gt_uninfected": img_missed_uninf,
            "missed_gt_parasitized": img_missed_par,
            "spurious_unmatched_detections": img_spurious,
            "detections": det_records
        })

        print(
            f"  [{idx:02d}/25] {fname} (Patient {patient_id}) | GPU: {infer_ms:5.1f}ms | "
            f"GT RBCs: {img_gt_uninf+img_gt_par:3d} (Par:{img_gt_par:2d}) | "
            f"Matched: {img_matched_uninf+img_matched_par:3d} (Par:{img_matched_par:2d}, Missed Par:{img_missed_par:2d}) | "
            f"Spurious: {img_spurious:2d}",
            flush=True
        )

    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    free_after, _ = torch.cuda.mem_get_info(0)
    stage1_elapsed = round(time.time() - t0, 2)

    np.savez_compressed(
        STAGE1_CROPS_NPZ,
        raw_crops=np.stack(all_raw_crops_224, axis=0),
        masked_crops=np.stack(all_masked_crops_224, axis=0)
    )

    stage1_summary = {
        "hardware": {
            "python_env": VENV_YOLO_GPU_PYTHON,
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "gpu_device_id": 0,
            "gpu_device_name": device_name,
            "total_vram_mb": round(total_vram / (1024 ** 2), 1),
            "free_vram_before_mb": round(free_before / (1024 ** 2), 1),
            "free_vram_after_mb": round(free_after / (1024 ** 2), 1),
            "peak_allocated_vram_mb": round(peak_vram_mb, 2),
            "batch_size": 1
        },
        "stage1_test_totals": {
            "test_images_count": len(test_files),
            "test_patients_count": 5,
            "gt_uninfected_rbc": total_gt_uninf,
            "gt_parasitized_rbc": total_gt_par,
            "gt_total_rbc": total_gt_uninf + total_gt_par,
            "gt_wbc": total_gt_wbc,
            "total_yolo_candidate_rbcs": global_crop_idx,
            "matched_gt_uninfected": total_matched_uninf,
            "matched_gt_parasitized": total_matched_par,
            "matched_gt_total_rbc": total_matched_uninf + total_matched_par,
            "missed_gt_uninfected": total_missed_uninf,
            "missed_gt_parasitized": total_missed_par,
            "missed_gt_total_rbc": total_missed_uninf + total_missed_par,
            "spurious_unmatched_detections": total_spurious_dets,
            "stage1_uninfected_localization_recall": round(total_matched_uninf / float(total_gt_uninf), 6),
            "stage1_parasitized_localization_recall": round(total_matched_par / float(total_gt_par), 6),
            "stage1_overall_rbc_localization_recall": round(
                (total_matched_uninf + total_matched_par) / float(total_gt_uninf + total_gt_par), 6
            ),
            "total_stage1_seconds": stage1_elapsed
        },
        "per_image": per_image_records
    }

    with open(STAGE1_JSON, "w", encoding="utf-8") as jf:
        json.dump(stage1_summary, jf, indent=2)

    print(
        f"\n[Stage 1 GPU Complete in {stage1_elapsed}s | Peak VRAM: {peak_vram_mb:.1f} MB] "
        f"Matched {total_matched_uninf+total_matched_par}/{total_gt_uninf+total_gt_par} GT RBCs "
        f"({(total_matched_uninf+total_matched_par)/(total_gt_uninf+total_gt_par)*100:.2f}%), "
        f"Missed Par: {total_missed_par}/{total_gt_par}, Missed Uninf: {total_missed_uninf}/{total_gt_uninf}, "
        f"Spurious: {total_spurious_dets}",
        flush=True
    )


def run_stage2_tf():
    """Runs in venv (TensorFlow CPU): classifies all Stage 1 candidate crops under Strategies A, B, C, D."""
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow import keras

    t0 = time.time()
    with open(STAGE1_JSON, "r", encoding="utf-8") as f:
        s1 = json.load(f)

    npz = np.load(STAGE1_CROPS_NPZ)
    raw_crops = npz["raw_crops"]
    masked_crops = npz["masked_crops"]

    print("\n" + "=" * 80, flush=True)
    print(f"STAGE 2 END-TO-END CLASSIFICATION ON {len(raw_crops)} YOLO CANDIDATE CROPS (CPU)", flush=True)
    print("=" * 80, flush=True)

    orig_model = keras.models.load_model(ORIG_CNN_PATH)
    ft_raw_model = keras.models.load_model(FINETUNED_RAW_PATH)
    ft_masked_model = keras.models.load_model(FINETUNED_MASKED_PATH)

    # Build frozen Block 1-3 extractor and top-layer models once, then process in 512-crop chunks
    feat_extractor = keras.Model(inputs=orig_model.inputs, outputs=orig_model.layers[10].output)

    def build_head(full_mdl):
        top_in = keras.Input(shape=(28, 28, 128))
        x = top_in
        for idx_l in range(11, len(full_mdl.layers)):
            lyr = full_mdl.layers[idx_l]
            if isinstance(lyr, keras.layers.BatchNormalization):
                x = lyr(x, training=False)
            else:
                x = lyr(x)
        return keras.Model(inputs=top_in, outputs=x)

    head_orig = build_head(orig_model)
    head_ft_raw = build_head(ft_raw_model)
    head_ft_masked = build_head(ft_masked_model)

    probs_A, probs_C = [], []
    probs_B, probs_D = [], []
    chunk = 512
    for i in range(0, len(raw_crops), chunk):
        f_r = feat_extractor.predict(raw_crops[i:i + chunk], batch_size=64, verbose=0)
        probs_A.append(head_orig.predict(f_r, batch_size=128, verbose=0).flatten())
        probs_C.append(head_ft_raw.predict(f_r, batch_size=128, verbose=0).flatten())

        f_m = feat_extractor.predict(masked_crops[i:i + chunk], batch_size=64, verbose=0)
        probs_B.append(head_orig.predict(f_m, batch_size=128, verbose=0).flatten())
        probs_D.append(head_ft_masked.predict(f_m, batch_size=128, verbose=0).flatten())

    probs_map = {
        "A_Orig_CNN_Raw": np.concatenate(probs_A, axis=0),
        "B_Orig_CNN_Masked": np.concatenate(probs_B, axis=0),
        "C_Finetuned_CNN_Raw": np.concatenate(probs_C, axis=0),
        "D_Finetuned_CNN_Masked": np.concatenate(probs_D, axis=0)
    }

    strat_titles = {
        "A_Orig_CNN_Raw": "Strategy A: Existing CNN (Raw Crops)",
        "B_Orig_CNN_Masked": "Strategy B: Existing CNN (Elliptical-Masked Crops)",
        "C_Finetuned_CNN_Raw": "Strategy C: Fine-Tuned CNN (Raw Crops)",
        "D_Finetuned_CNN_Masked": "Strategy D: Fine-Tuned CNN (Elliptical-Masked Crops)"
    }

    end_to_end_results = {}
    csv_rows = []

    gt_tot_uninf = s1["stage1_test_totals"]["gt_uninfected_rbc"]
    gt_tot_par = s1["stage1_test_totals"]["gt_parasitized_rbc"]
    missed_tot_uninf = s1["stage1_test_totals"]["missed_gt_uninfected"]
    missed_tot_par = s1["stage1_test_totals"]["missed_gt_parasitized"]

    for sk, s_title in strat_titles.items():
        probs = probs_map[sk]
        preds = (probs >= 0.50).astype(int)

        tp_matched = 0
        fp_matched_uninf = 0
        tn_matched = 0
        fn_matched_par = 0
        fp_spurious = 0
        tn_spurious = 0

        img_abs_errors = []
        patient_agg = defaultdict(lambda: {
            "gt_uninf": 0, "gt_par": 0,
            "missed_uninf": 0, "missed_par": 0,
            "yolo_candidates": 0, "cnn_pred_par": 0,
            "tp": 0, "fp_matched": 0, "fp_spurious": 0, "tn": 0, "fn_cnn": 0
        })

        for img_rec in s1["per_image"]:
            pid = img_rec["patient_id"]
            p_agg = patient_agg[pid]
            p_agg["gt_uninf"] += img_rec["gt_uninfected_rbc"]
            p_agg["gt_par"] += img_rec["gt_parasitized_rbc"]
            p_agg["missed_uninf"] += img_rec["missed_gt_uninfected"]
            p_agg["missed_par"] += img_rec["missed_gt_parasitized"]
            p_agg["yolo_candidates"] += img_rec["yolo_valid_rbc_candidates"]

            img_pred_par = 0
            for det in img_rec["detections"]:
                gidx = det["global_crop_idx"]
                pr = int(preds[gidx])
                gt_c = det["matched_gt_class_id"]
                if pr == 1:
                    img_pred_par += 1
                    p_agg["cnn_pred_par"] += 1

                if gt_c == 1:
                    if pr == 1:
                        tp_matched += 1
                        p_agg["tp"] += 1
                    else:
                        fn_matched_par += 1
                        p_agg["fn_cnn"] += 1
                elif gt_c == 0:
                    if pr == 1:
                        fp_matched_uninf += 1
                        p_agg["fp_matched"] += 1
                    else:
                        tn_matched += 1
                        p_agg["tn"] += 1
                else:
                    if pr == 1:
                        fp_spurious += 1
                        p_agg["fp_spurious"] += 1
                    else:
                        tn_spurious += 1

            gt_img_par_pct = (img_rec["gt_parasitized_rbc"] / float(img_rec["gt_total_rbc"]) * 100.0)
            pred_img_par_pct = (img_pred_par / float(img_rec["yolo_valid_rbc_candidates"]) * 100.0) if img_rec["yolo_valid_rbc_candidates"] > 0 else 0.0
            abs_err = abs(pred_img_par_pct - gt_img_par_pct)
            img_abs_errors.append(abs_err)

            if sk == "C_Finetuned_CNN_Raw":
                # We will populate one consolidated row per image across all strategies below
                pass

        # End-to-End Cell Accounting (including Stage 1 missed Parasitized RBCs as FN!)
        total_fn_e2e = fn_matched_par + missed_tot_par
        total_fp_e2e = fp_matched_uninf + fp_spurious

        # Sensitivity against ALL 376 ground-truth Parasitized RBCs in the test set:
        e2e_par_recall = tp_matched / float(gt_tot_par) if gt_tot_par > 0 else 0.0
        # Precision across all predicted Parasitized cells (including spurious boxes):
        e2e_par_prec = tp_matched / float(tp_matched + total_fp_e2e) if (tp_matched + total_fp_e2e) > 0 else 0.0
        e2e_par_f1 = (
            2 * e2e_par_prec * e2e_par_recall / (e2e_par_prec + e2e_par_recall)
            if (e2e_par_prec + e2e_par_recall) > 0 else 0.0
        )
        # Specificity on matched GT Uninfected RBCs (4,335 matched) and against all 4,418 GT Uninfected RBCs:
        matched_uninf_spec = tn_matched / float(tn_matched + fp_matched_uninf) if (tn_matched + fp_matched_uninf) > 0 else 0.0
        strict_uninf_recall = tn_matched / float(gt_tot_uninf) if gt_tot_uninf > 0 else 0.0

        # Per-patient end-to-end summary
        per_patient_out = {}
        pat_abs_errors = []
        for pid in sorted(patient_agg.keys()):
            pa = patient_agg[pid]
            gt_rbc_p = pa["gt_uninf"] + pa["gt_par"]
            gt_pct = pa["gt_par"] / float(gt_rbc_p) * 100.0
            pred_pct = pa["cnn_pred_par"] / float(pa["yolo_candidates"]) * 100.0
            p_err = abs(pred_pct - gt_pct)
            pat_abs_errors.append(p_err)
            tot_fn_p = pa["fn_cnn"] + pa["missed_par"]
            tot_fp_p = pa["fp_matched"] + pa["fp_spurious"]
            p_sens_e2e = pa["tp"] / float(pa["gt_par"]) if pa["gt_par"] > 0 else 0.0
            p_spec_matched = pa["tn"] / float(pa["tn"] + pa["fp_matched"]) if (pa["tn"] + pa["fp_matched"]) > 0 else 0.0
            per_patient_out[pid] = {
                "gt_total_rbc": gt_rbc_p,
                "gt_uninfected": pa["gt_uninf"],
                "gt_parasitized": pa["gt_par"],
                "gt_parasitemia_percent": round(gt_pct, 4),
                "yolo_candidate_rbcs": pa["yolo_candidates"],
                "stage1_missed_parasitized": pa["missed_par"],
                "stage1_missed_uninfected": pa["missed_uninf"],
                "cnn_predicted_parasitized": pa["cnn_pred_par"],
                "estimated_parasitemia_percent": round(pred_pct, 4),
                "parasitemia_abs_error_pct": round(p_err, 4),
                "tp": pa["tp"],
                "fp_total": tot_fp_p,
                "fp_on_gt_uninfected": pa["fp_matched"],
                "fp_on_spurious_boxes": pa["fp_spurious"],
                "tn_matched": pa["tn"],
                "fn_cnn_on_matched": pa["fn_cnn"],
                "fn_total_including_stage1_misses": tot_fn_p,
                "end_to_end_parasitized_sensitivity": round(p_sens_e2e, 6),
                "matched_uninfected_specificity": round(p_spec_matched, 6)
            }

        end_to_end_results[sk] = {
            "strategy_label": s_title,
            "tp": tp_matched,
            "fp_on_matched_uninfected": fp_matched_uninf,
            "fp_on_spurious_detections": fp_spurious,
            "fp_total_end_to_end": total_fp_e2e,
            "tn_on_matched_uninfected": tn_matched,
            "tn_on_spurious_detections": tn_spurious,
            "fn_stage2_cnn_on_matched": fn_matched_par,
            "fn_stage1_yolo_missed_parasitized": missed_tot_par,
            "fn_total_end_to_end": total_fn_e2e,
            "end_to_end_parasitized_precision": round(e2e_par_prec, 6),
            "end_to_end_parasitized_sensitivity_recall": round(e2e_par_recall, 6),
            "end_to_end_parasitized_f1": round(e2e_par_f1, 6),
            "matched_uninfected_specificity": round(matched_uninf_spec, 6),
            "strict_uninfected_localization_and_specificity": round(strict_uninf_recall, 6),
            "image_level_parasitemia_mae_percent": round(float(np.mean(img_abs_errors)), 4),
            "patient_level_parasitemia_mae_percent": round(float(np.mean(pat_abs_errors)), 4),
            "per_patient": per_patient_out
        }

    # Build per-image comparison CSV across all 25 test images
    for img_rec in s1["per_image"]:
        row = {
            "image_filename": img_rec["image_filename"],
            "patient_id": img_rec["patient_id"],
            "gt_total_rbc": img_rec["gt_total_rbc"],
            "gt_uninfected_rbc": img_rec["gt_uninfected_rbc"],
            "gt_parasitized_rbc": img_rec["gt_parasitized_rbc"],
            "gt_parasitemia_pct": round(img_rec["gt_parasitized_rbc"] / float(img_rec["gt_total_rbc"]) * 100.0, 2),
            "yolo_valid_rbcs": img_rec["yolo_valid_rbc_candidates"],
            "stage1_missed_par": img_rec["missed_gt_parasitized"],
            "stage1_missed_uninf": img_rec["missed_gt_uninfected"]
        }
        g_indices = [d["global_crop_idx"] for d in img_rec["detections"]]
        n_cand = len(g_indices)
        for sk, short_tag in [
            ("A_Orig_CNN_Raw", "stratA_orig_raw"),
            ("B_Orig_CNN_Masked", "stratB_orig_masked"),
            ("C_Finetuned_CNN_Raw", "stratC_ft_raw"),
            ("D_Finetuned_CNN_Masked", "stratD_ft_masked")
        ]:
            n_par = int(np.sum(probs_map[sk][g_indices] >= 0.50))
            pct = round(n_par / float(n_cand) * 100.0, 2) if n_cand > 0 else 0.0
            row[f"{short_tag}_par_count"] = n_par
            row[f"{short_tag}_par_pct"] = pct
        csv_rows.append(row)

    with open(PER_IMAGE_CSV_PATH, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    stage2_elapsed = round(time.time() - t0, 2)

    full_report = {
        "phase": "Phase 6 — Full Two-Stage Pipeline End-to-End Evaluation (Including Detection Misses)",
        "stage1_gpu_summary": s1["hardware"],
        "stage1_detection_accounting": s1["stage1_test_totals"],
        "stage2_cpu_processing_seconds": stage2_elapsed,
        "end_to_end_strategies_comparison": end_to_end_results,
        "per_image_comparison_csv": PER_IMAGE_CSV_PATH
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(full_report, jf, indent=2)

    st_tot = s1["stage1_test_totals"]
    txt_lines = [
        "=" * 96,
        "SMARTCELL PHASE 6: END-TO-END TWO-STAGE PIPELINE EVALUATION (INCLUDING DETECTION MISSES)",
        "=" * 96,
        f"Stage 1 Hardware: {s1['hardware']['gpu_device_name']} (PyTorch {s1['hardware']['pytorch_version']}, device=0, Peak VRAM: {s1['hardware']['peak_allocated_vram_mb']} MB)",
        f"Stage 2 Hardware: CPU (D:\\smartcell\\venv, TensorFlow 2.21.0 native Windows)",
        "",
        "1. STAGE 1 (YOLOv8n) CELL LOCALIZATION & MISS ACCOUNTING ACROSS 25 TEST IMAGES (5 PATIENTS):",
        f"   Ground-Truth RBCs:          {st_tot['gt_total_rbc']} (Uninfected: {st_tot['gt_uninfected_rbc']}, Parasitized: {st_tot['gt_parasitized_rbc']}; Excluded GT WBCs: {st_tot['gt_wbc']})",
        f"   Matched Ground-Truth RBCs:  {st_tot['matched_gt_total_rbc']} / {st_tot['gt_total_rbc']} ({st_tot['stage1_overall_rbc_localization_recall']*100:.2f}%)",
        f"     - Matched Uninfected:     {st_tot['matched_gt_uninfected']} / {st_tot['gt_uninfected_rbc']} ({st_tot['stage1_uninfected_localization_recall']*100:.2f}%) | Missed: {st_tot['missed_gt_uninfected']}",
        f"     - Matched Parasitized:    {st_tot['matched_gt_parasitized']} / {st_tot['gt_parasitized_rbc']} ({st_tot['stage1_parasitized_localization_recall']*100:.2f}%) | Missed: {st_tot['missed_gt_parasitized']} (Counted as E2E FN!)",
        f"   Spurious / Unmatched Boxes: {st_tot['spurious_unmatched_detections']} (Included in E2E FP accounting)",
        "",
        "2. END-TO-END TWO-STAGE PERFORMANCE (PENALIZING STAGE 1 MISSES AS FALSE NEGATIVES):",
        "-" * 96,
        f"{'Strategy':<42} | {'E2E Prec':>8} | {'E2E Sens':>8} | {'E2E F1':>7} | {'Spec(M)':>7} | {'TP':>3} {'FP':>4} {'FN_CNN':>6} {'FN_Miss':>7} {'FN_Tot':>6} | {'Img MAE':>7} {'Pat MAE':>7}",
        "-" * 96
    ]
    for sk, r in end_to_end_results.items():
        txt_lines.append(
            f"{r['strategy_label']:<42} | {r['end_to_end_parasitized_precision']*100:7.2f}% | "
            f"{r['end_to_end_parasitized_sensitivity_recall']*100:7.2f}% | {r['end_to_end_parasitized_f1']:7.4f} | "
            f"{r['matched_uninfected_specificity']*100:6.2f}% | "
            f"{r['tp']:3d} {r['fp_total_end_to_end']:4d} {r['fn_stage2_cnn_on_matched']:6d} "
            f"{r['fn_stage1_yolo_missed_parasitized']:7d} {r['fn_total_end_to_end']:6d} | "
            f"{r['image_level_parasitemia_mae_percent']:6.2f}% {r['patient_level_parasitemia_mae_percent']:6.2f}%"
        )
    txt_lines.extend([
        "-" * 96,
        "",
        "3. END-TO-END PER-PATIENT PARASITEMIA ESTIMATION ACROSS THE 5 HELD-OUT TEST PATIENTS:"
    ])
    for sk, r in end_to_end_results.items():
        txt_lines.append(f"\n  [{r['strategy_label']}]")
        for pid, pm in r["per_patient"].items():
            txt_lines.append(
                f"    Patient {pid:<20} | GT Par: {pm['gt_parasitized']:3d}/{pm['gt_total_rbc']:4d} ({pm['gt_parasitemia_percent']:5.2f}%) | "
                f"Est Par: {pm['cnn_predicted_parasitized']:3d}/{pm['yolo_candidate_rbcs']:4d} ({pm['estimated_parasitemia_percent']:5.2f}%, Err: {pm['parasitemia_abs_error_pct']:5.2f}%) | "
                f"TP={pm['tp']:3d} FP={pm['fp_total']:3d} FN_CNN={pm['fn_cnn_on_matched']:2d} FN_Miss={pm['stage1_missed_parasitized']:2d} | "
                f"E2E Sens={pm['end_to_end_parasitized_sensitivity']*100:5.1f}%"
            )
    txt_lines.append("=" * 96)
    txt_out = "\n".join(txt_lines) + "\n"
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as tf_out:
        tf_out.write(txt_out)

    print("\n" + txt_out, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-gpu-only", action="store_true")
    parser.add_argument("--stage2-tf-only", action="store_true")
    args = parser.parse_args()

    if args.stage1_gpu_only:
        run_stage1_gpu()
        return
    if args.stage2_tf_only:
        run_stage2_tf()
        return

    subprocess.run([VENV_YOLO_GPU_PYTHON, "-u", os.path.abspath(__file__), "--stage1-gpu-only"], check=True)
    subprocess.run([VENV_TF_PYTHON, "-u", os.path.abspath(__file__), "--stage2-tf-only"], check=True)


if __name__ == "__main__":
    main()
