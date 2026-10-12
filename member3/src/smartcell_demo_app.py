r"""
smartcell_demo_app.py
---------------------
SmartCell Phase 7: Interactive Local Demonstration Application
Around the Verified Two-Stage YOLOv8n (GPU) + Custom CNN (CPU) Blood-Smear Pipeline.

DISCLAIMER:
SmartCell is an experimental academic research prototype developed for a college
engineering project. It is NOT a clinically validated diagnostic device.
Nor does evaluation on the 25-image NIH-NLM test split establish independent
clinical generalization.

Features:
1. Full-resolution microscope blood-smear image upload (.jpg, .jpeg, .png, .bmp, .tif, .tiff)
   plus quick-selection of the 25 held-out NIH-NLM test-set images for live demonstration.
2. Selection between:
   - Original CNN (`D:\smartcell\models\malaria_cnn_best.keras`)
   - Fine-Tuned Raw CNN (`D:\smartcell\models\malaria_cnn_finetuned_raw_best.keras`)
   - Fine-Tuned Masked CNN (`D:\smartcell\models\malaria_cnn_finetuned_masked_best.keras`)
3. Validated crop mode selection (`raw` rectangular vs `masked` elliptical black-corner)
   and configurable inference thresholds (YOLO conf, YOLO NMS IoU, dedup IoU, CNN threshold,
   crop padding, minimum box size).
4. Runs Stage 1 YOLOv8n detection in `D:\smartcell\venv_yolo_gpu` on NVIDIA GeForce RTX 2050
   (CUDA device 0) with graceful fallback to CPU if GPU is unavailable.
5. Runs Stage 2 CNN classification in `D:\smartcell\venv` (TensorFlow 2.21.0) without modifying
   any environment dependencies.
6. Displays the annotated full-smear image, RBC crop contact sheet, individual cell crop
   thumbnails, and separate counts for:
   - Predicted Parasitized RBCs
   - Predicted Uninfected RBCs
   - Excluded WBC detections
   - Rejected / duplicate detections
7. Displays the experimental parasitemia estimate, model confidence statistics
   (mean/median/min/max for YOLO and CNN), hardware telemetry, and processing time breakdown.
8. Provides one-click CSV, JSON, TXT, and annotated JPG export downloads.
9. Gracefully handles invalid/corrupted images, zero detections, inference errors, and
   unavailable GPU.

Usage:
  D:\smartcell\venv\Scripts\python.exe D:\smartcell\scripts\smartcell_demo_app.py --port 8501
"""

import os
import sys
import csv
import json
import time
import uuid
import shutil
import base64
import mimetypes
import argparse
import traceback
import subprocess
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BASE_DIR = r"D:\smartcell"
VENV_TF_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
VENV_YOLO_GPU_PYTHON = os.path.join(BASE_DIR, "venv_yolo_gpu", "Scripts", "python.exe")
VENV_YOLO_CPU_PYTHON = os.path.join(BASE_DIR, "venv_yolo", "Scripts", "python.exe")

PIPELINE_SCRIPT = os.path.join(BASE_DIR, "scripts", "run_smartcell_pipeline.py")
DEMO_RUNS_ROOT = os.path.join(BASE_DIR, "results", "demo_app_runs")
UPLOADS_DIR = os.path.join(DEMO_RUNS_ROOT, "_uploads")

TEST_IMAGES_DIR = os.path.join(BASE_DIR, "yolo_malaria", "images", "test")
TEST_LABELS_DIR = os.path.join(BASE_DIR, "yolo_malaria", "labels", "test")
POLYGON_SET_DIR = os.path.join(BASE_DIR, "Polygon Set")

MODELS_MAP = {
    "finetuned_raw": {
        "label": "Fine-Tuned Raw CNN (Phase 6 Domain-Adapted — Recommended)",
        "path": os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_raw_best.keras"),
        "recommended_crop_mode": "raw",
        "description": "Fine-tuned on 24,112 native rectangular full-smear crops (Test Acc: 98.35%, Par F1: 0.9016)."
    },
    "finetuned_masked": {
        "label": "Fine-Tuned Masked CNN (Phase 6 Elliptical-Masked)",
        "path": os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_masked_best.keras"),
        "recommended_crop_mode": "masked",
        "description": "Validated on elliptical black-corner masked full-smear crops (Test Acc: 98.10%, Par F1: 0.8905)."
    },
    "original": {
        "label": "Original Baseline CNN (Phase 1 Single-Cell Dataset)",
        "path": os.path.join(BASE_DIR, "models", "malaria_cnn_best.keras"),
        "recommended_crop_mode": "masked",
        "description": "Trained on 27,558 black-background NIH single-cell crops (95.04% single-cell Acc; sensitive to raw corners)."
    }
}

YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "results", "yolo_malaria_run", "weights", "best.pt")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Cached hardware status so UI loads instantaneously
_CACHED_ENV_STATUS = None


def build_image_to_patient_map():
    mapping = {}
    if os.path.exists(POLYGON_SET_DIR):
        for pat in sorted(os.listdir(POLYGON_SET_DIR)):
            img_dir = os.path.join(POLYGON_SET_DIR, pat, "Img")
            if os.path.isdir(img_dir):
                for fn in os.listdir(img_dir):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        mapping[fn] = pat
    return mapping


IMAGE_TO_PATIENT = build_image_to_patient_map()


def inspect_ground_truth_for_stem(stem):
    """Checks if a microscope image stem has a known ground-truth label file in yolo_malaria/labels."""
    for split in ("test", "val", "train"):
        lbl_path = os.path.join(BASE_DIR, "yolo_malaria", "labels", split, f"{stem}.txt")
        if os.path.exists(lbl_path):
            uninf = 0
            par = 0
            wbc = 0
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid == 0:
                        uninf += 1
                    elif cid == 1:
                        par += 1
                    elif cid == 2:
                        wbc += 1
            tot_rbc = uninf + par
            par_pct = round((par / float(tot_rbc)) * 100.0, 2) if tot_rbc > 0 else 0.0
            return {
                "has_ground_truth": True,
                "split": split,
                "patient_id": IMAGE_TO_PATIENT.get(f"{stem}.jpg", "Unknown"),
                "gt_uninfected_rbc": uninf,
                "gt_parasitized_rbc": par,
                "gt_wbc": wbc,
                "gt_total_rbc": tot_rbc,
                "gt_parasitemia_percent": par_pct
            }
    return {"has_ground_truth": False}


def get_preset_test_images():
    """Returns metadata for the 25 held-out test set images for quick demo selection."""
    presets = []
    if not os.path.exists(TEST_IMAGES_DIR):
        return presets
    for fn in sorted(os.listdir(TEST_IMAGES_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            stem = os.path.splitext(fn)[0]
            gt = inspect_ground_truth_for_stem(stem)
            presets.append({
                "filename": fn,
                "stem": stem,
                "patient_id": gt.get("patient_id", "Unknown"),
                "gt_uninfected_rbc": gt.get("gt_uninfected_rbc", 0),
                "gt_parasitized_rbc": gt.get("gt_parasitized_rbc", 0),
                "gt_wbc": gt.get("gt_wbc", 0),
                "gt_total_rbc": gt.get("gt_total_rbc", 0),
                "gt_parasitemia_percent": gt.get("gt_parasitemia_percent", 0.0)
            })
    return presets


def query_environment_status(force_refresh=False):
    """Queries GPU availability in venv_yolo_gpu and TensorFlow/model readiness in venv."""
    global _CACHED_ENV_STATUS
    if _CACHED_ENV_STATUS is not None and not force_refresh:
        return _CACHED_ENV_STATUS

    gpu_info = {
        "venv_yolo_gpu_exists": os.path.exists(VENV_YOLO_GPU_PYTHON),
        "venv_yolo_cpu_exists": os.path.exists(VENV_YOLO_CPU_PYTHON),
        "venv_tf_exists": os.path.exists(VENV_TF_PYTHON),
        "cuda_available": False,
        "gpu_name": "Unavailable (CPU Fallback Active)",
        "pytorch_version": "Unknown",
        "free_vram_mb": 0.0,
        "total_vram_mb": 0.0,
        "yolo_checkpoint_exists": os.path.exists(YOLO_WEIGHTS_PATH),
        "yolo_checkpoint_path": YOLO_WEIGHTS_PATH
    }

    if os.path.exists(VENV_YOLO_GPU_PYTHON):
        try:
            probe_code = (
                "import json, torch; "
                "avail = torch.cuda.is_available(); "
                "free_b, tot_b = torch.cuda.mem_get_info(0) if avail else (0, 0); "
                "print(json.dumps({"
                "'cuda_available': avail, "
                "'gpu_name': torch.cuda.get_device_name(0) if avail else 'CPU', "
                "'pytorch_version': torch.__version__, "
                "'free_vram_mb': round(free_b / (1024**2), 1), "
                "'total_vram_mb': round(tot_b / (1024**2), 1)"
                "}))"
            )
            res = subprocess.run(
                [VENV_YOLO_GPU_PYTHON, "-c", probe_code],
                capture_output=True, text=True, timeout=15
            )
            if res.returncode == 0 and res.stdout.strip():
                parsed = json.loads(res.stdout.strip().splitlines()[-1])
                gpu_info.update(parsed)
        except Exception as e:
            gpu_info["probe_error"] = str(e)

    models_status = {}
    for k, meta in MODELS_MAP.items():
        models_status[k] = {
            "variant": k,
            "label": meta["label"],
            "path": meta["path"],
            "exists": os.path.exists(meta["path"]),
            "recommended_crop_mode": meta["recommended_crop_mode"],
            "description": meta["description"]
        }

    _CACHED_ENV_STATUS = {
        "timestamp": datetime.now().isoformat(),
        "gpu_environment": gpu_info,
        "cnn_models": models_status,
        "preset_test_images": get_preset_test_images()
    }
    return _CACHED_ENV_STATUS


def validate_image_file(filepath):
    """Validates that a file is a readable, non-corrupted RGB/grayscale image."""
    from PIL import Image
    if not os.path.exists(filepath):
        return False, "Uploaded image file does not exist on disk.", None
    if os.path.getsize(filepath) == 0:
        return False, "Uploaded file is empty (0 bytes).", None

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", None

    try:
        with Image.open(filepath) as im:
            im.verify()
        with Image.open(filepath) as im2:
            rgb = im2.convert("RGB")
            w, h = rgb.size
            if w < 32 or h < 32:
                return False, f"Image dimensions ({w}x{h} px) are too small for blood-smear analysis (minimum 32x32 px).", None
            return True, "Valid image", {"width": w, "height": h}
    except Exception as e:
        return False, f"Invalid or corrupted image file: {type(e).__name__}: {e}", None


def execute_pipeline_run(
    image_path,
    cnn_variant="finetuned_raw",
    crop_mode="raw",
    yolo_device="0",
    yolo_conf=0.25,
    yolo_iou=0.45,
    dedup_iou=0.50,
    cnn_thresh=0.50,
    pad_px=4,
    min_box_size=16,
    max_det=350,
    test_zero_detections=False,
    inject_test_invalid_boxes=False,
    run_id=None
):
    """
    Executes the verified two-stage pipeline (`run_smartcell_pipeline.py`) and returns
    the structured payload matching the saved JSON, CSV, and TXT reports on disk.
    """
    os.makedirs(DEMO_RUNS_ROOT, exist_ok=True)
    is_valid, err_msg, dim_info = validate_image_file(image_path)
    if not is_valid:
        return {
            "success": False,
            "error_type": "InvalidImageError",
            "error_message": err_msg
        }

    if cnn_variant not in MODELS_MAP:
        return {
            "success": False,
            "error_type": "InvalidParameterError",
            "error_message": f"Unknown CNN variant '{cnn_variant}'. Valid options: {list(MODELS_MAP.keys())}"
        }

    if crop_mode not in ("raw", "masked", "ellipse_masked"):
        return {
            "success": False,
            "error_type": "InvalidParameterError",
            "error_message": f"Unknown crop mode '{crop_mode}'. Valid options: raw, masked, ellipse_masked"
        }

    # Clamp numeric thresholds safely
    try:
        yolo_conf = max(0.01, min(0.99, float(yolo_conf)))
        yolo_iou = max(0.05, min(0.95, float(yolo_iou)))
        dedup_iou = max(0.05, min(0.99, float(dedup_iou)))
        cnn_thresh = max(0.01, min(0.99, float(cnn_thresh)))
        pad_px = max(0, min(32, int(pad_px)))
        min_box_size = max(4, min(128, int(min_box_size)))
        max_det = max(10, min(1000, int(max_det)))
    except Exception as e:
        return {
            "success": False,
            "error_type": "ThresholdValidationError",
            "error_message": f"Invalid numeric threshold parameter: {e}"
        }

    stem = os.path.splitext(os.path.basename(image_path))[0]
    if not run_id:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uid = uuid.uuid4().hex[:6]
        run_id = f"run_{ts}_{stem}_{cnn_variant}_{crop_mode}_{short_uid}"

    run_dir = os.path.join(DEMO_RUNS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    cmd = [
        VENV_TF_PYTHON,
        "-u",
        PIPELINE_SCRIPT,
        "--image", os.path.abspath(image_path),
        "--output-dir", os.path.abspath(run_dir),
        "--cnn-variant", cnn_variant,
        "--crop-mode", crop_mode,
        "--yolo-device", str(yolo_device),
        "--yolo-conf", str(yolo_conf),
        "--yolo-iou", str(yolo_iou),
        "--dedup-iou", str(dedup_iou),
        "--cnn-thresh", str(cnn_thresh),
        "--pad-px", str(pad_px),
        "--min-box-size", str(min_box_size),
        "--max-det", str(max_det)
    ]
    if test_zero_detections:
        cmd.append("--test-zero-detections")
    if inject_test_invalid_boxes:
        cmd.append("--inject-test-invalid-boxes")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error_type": "PipelineTimeoutError",
            "error_message": "Two-stage inference timed out after 300 seconds."
        }
    except Exception as e:
        return {
            "success": False,
            "error_type": "PipelineExecutionError",
            "error_message": f"Failed to launch two-stage pipeline: {e}"
        }

    if proc.returncode != 0:
        return {
            "success": False,
            "error_type": "InferenceSubprocessError",
            "error_message": f"Pipeline exited with code {proc.returncode}.",
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else ""
        }

    summary_json_path = os.path.join(run_dir, f"field_summary_{stem}.json")
    csv_path = os.path.join(run_dir, f"cell_predictions_{stem}.csv")
    rejected_json_path = os.path.join(run_dir, f"rejected_and_excluded_{stem}.json")
    summary_txt_path = os.path.join(run_dir, f"field_summary_{stem}.txt")
    annotated_jpg_path = os.path.join(run_dir, f"annotated_{stem}.jpg")
    contact_sheet_path = os.path.join(run_dir, f"contact_sheet_{stem}.jpg")

    if not os.path.exists(summary_json_path):
        return {
            "success": False,
            "error_type": "MissingOutputError",
            "error_message": f"Expected summary report not found at {summary_json_path}."
        }

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    with open(rejected_json_path, "r", encoding="utf-8") as f:
        rejected_data = json.load(f)

    txt_report = ""
    if os.path.exists(summary_txt_path):
        with open(summary_txt_path, "r", encoding="utf-8") as f:
            txt_report = f.read()

    cell_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cell_rows.append(r)

    # Check if ground-truth reference exists for this image stem
    gt_ref = inspect_ground_truth_for_stem(stem)

    return {
        "success": True,
        "run_id": run_id,
        "run_directory": run_dir,
        "stem": stem,
        "summary": summary_data,
        "rejected_and_excluded": rejected_data,
        "cell_predictions": cell_rows,
        "text_report": txt_report,
        "ground_truth_reference": gt_ref,
        "urls": {
            "annotated_image": f"/api/file?run_id={run_id}&file=annotated_{stem}.jpg" if os.path.exists(annotated_jpg_path) else None,
            "contact_sheet": f"/api/file?run_id={run_id}&file=contact_sheet_{stem}.jpg" if os.path.exists(contact_sheet_path) else None,
            "export_csv": f"/api/file?run_id={run_id}&file=cell_predictions_{stem}.csv&download=1",
            "export_summary_json": f"/api/file?run_id={run_id}&file=field_summary_{stem}.json&download=1",
            "export_rejected_json": f"/api/file?run_id={run_id}&file=rejected_and_excluded_{stem}.json&download=1",
            "export_txt": f"/api/file?run_id={run_id}&file=field_summary_{stem}.txt&download=1"
        }
    }


# ---------------------------------------------------------------------------
# Embedded Single-Page Web Application HTML/CSS/JS
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartCell — Two-Stage YOLOv8n + CNN Malaria Screening Prototype</title>
<style>
  :root {
    --bg: #0f141c;
    --panel: #171f2c;
    --panel-alt: #1e293b;
    --border: #2e3c52;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --danger: #ef4444;
    --danger-bg: rgba(239, 68, 68, 0.14);
    --success: #22c55e;
    --success-bg: rgba(34, 197, 94, 0.14);
    --warn: #f59e0b;
    --warn-bg: rgba(245, 158, 11, 0.14);
    --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  /* Prototype Disclaimer Banner */
  .disclaimer-banner {
    background: linear-gradient(90deg, #7f1d1d, #991b1b, #7f1d1d);
    color: #fef2f2;
    padding: 10px 24px;
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
    border-bottom: 1px solid #ef4444;
    letter-spacing: 0.2px;
  }
  .header {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 16px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }
  .brand h1 {
    font-size: 1.35rem;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand p {
    font-size: 0.84rem;
    color: var(--muted);
  }
  .hw-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .badge {
    font-size: 0.78rem;
    padding: 5px 11px;
    border-radius: 6px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    font-weight: 600;
  }
  .badge.gpu-ok { border-color: var(--success); color: #86efac; }
  .badge.cpu-ok { border-color: var(--accent); color: #7dd3fc; }
  .layout {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 20px;
    max-width: 1720px;
    margin: 20px auto;
    padding: 0 20px 40px;
  }
  @media (max-width: 1100px) {
    .layout { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 18px;
  }
  .card-title {
    font-size: 0.96rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  label {
    display: block;
    font-size: 0.82rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 5px;
    margin-top: 12px;
  }
  select, input[type="number"], input[type="text"], input[type="file"] {
    width: 100%;
    padding: 9px 11px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 0.86rem;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .hint {
    font-size: 0.76rem;
    color: var(--muted);
    margin-top: 4px;
  }
  .btn-primary {
    width: 100%;
    margin-top: 16px;
    padding: 12px;
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .btn-primary:hover { opacity: 0.92; }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-export {
    display: inline-block;
    padding: 7px 13px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    text-decoration: none;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 6px;
  }
  .btn-export:hover { border-color: var(--accent); color: #fff; }
  /* KPI Grid */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
  }
  .kpi-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 14px;
  }
  .kpi-box .kpi-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
    font-weight: 600;
  }
  .kpi-box .kpi-val {
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 4px;
  }
  .kpi-box .kpi-sub {
    font-size: 0.76rem;
    color: var(--muted);
    margin-top: 3px;
  }
  .kpi-par { border-left: 4px solid var(--danger); }
  .kpi-par .kpi-val { color: #f87171; }
  .kpi-uninf { border-left: 4px solid var(--success); }
  .kpi-uninf .kpi-val { color: #4ade80; }
  .kpi-wbc { border-left: 4px solid var(--warn); }
  .kpi-wbc .kpi-val { color: #fbbf24; }
  .kpi-rej { border-left: 4px solid var(--purple); }
  .kpi-rej .kpi-val { color: #c084fc; }
  .kpi-rate { border-left: 4px solid var(--accent); }
  .kpi-rate .kpi-val { color: #38bdf8; }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .tab-btn {
    padding: 8px 15px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--muted);
    font-size: 0.86rem;
    font-weight: 600;
    cursor: pointer;
  }
  .tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  .img-viewer {
    width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: #090d14;
    max-height: 740px;
    object-fit: contain;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
  }
  th, td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    text-align: left;
  }
  th {
    background: var(--panel-alt);
    color: #cbd5e1;
    font-weight: 600;
    position: sticky;
    top: 0;
  }
  .pill {
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
  }
  .pill-par { background: var(--danger-bg); color: #fca5a5; border: 1px solid var(--danger); }
  .pill-uninf { background: var(--success-bg); color: #86efac; border: 1px solid var(--success); }
  .alert-error {
    background: var(--danger-bg);
    border: 1px solid var(--danger);
    color: #fecaca;
    padding: 14px;
    border-radius: 8px;
    margin-bottom: 16px;
  }
  .alert-info {
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid var(--accent);
    color: #bae6fd;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 0.85rem;
  }
  pre.report-box {
    background: #090d14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    font-family: Consolas, Monaco, monospace;
    font-size: 0.8rem;
    overflow-x: auto;
    color: #cbd5e1;
    white-space: pre-wrap;
  }
  .crop-thumb {
    width: 52px;
    height: 52px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 0.81rem;
    color: var(--muted);
  }
  .checkbox-row input { width: auto; }
</style>
</head>
<body>

<div class="disclaimer-banner">
  RESEARCH PROTOTYPE NOTICE: SmartCell is an experimental academic research prototype for a college engineering demonstration.
  It is NOT a clinically validated diagnostic device and must NEVER be used for clinical diagnosis or patient care.
</div>

<header class="header">
  <div class="brand">
    <h1>SmartCell — Two-Stage YOLOv8n + CNN Malaria Blood-Smear Analyzer</h1>
    <p>Stage 1: YOLOv8n Candidate RBC Detection (GPU CUDA:0) &rarr; Stage 2: Domain-Adapted CNN Classification (TensorFlow CPU)</p>
  </div>
  <div class="hw-badges" id="hwBadges">
    <span class="badge">Checking GPU &amp; Environment...</span>
  </div>
</header>

<main class="layout">
  <!-- Left Control Panel -->
  <section>
    <div class="card">
      <div class="card-title">1. Blood-Smear Image Input</div>
      <label for="imageUpload">Upload Full-Resolution Microscope Smear (.jpg, .png, .tif)</label>
      <input type="file" id="imageUpload" accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff">
      <div class="hint">Or select a held-out NIH-NLM test-set blood smear below:</div>

      <label for="presetSelect">Preset Held-Out Test Smear (25 Test Images, 5 Patients)</label>
      <select id="presetSelect">
        <option value="">-- Use Uploaded Image Above --</option>
      </select>
    </div>

    <div class="card">
      <div class="card-title">2. Stage 2 Classifier &amp; Crop Mode</div>
      <label for="cnnVariant">CNN Model Checkpoint</label>
      <select id="cnnVariant" onchange="onVariantChange()">
        <option value="finetuned_raw" selected>Fine-Tuned Raw CNN (Phase 6 Domain-Adapted — Recommended)</option>
        <option value="finetuned_masked">Fine-Tuned Masked CNN (Phase 6 Elliptical-Masked)</option>
        <option value="original">Original Baseline CNN (Phase 1 Single-Cell Dataset)</option>
      </select>
      <div class="hint" id="variantDesc">Fine-tuned on 24,112 native rectangular full-smear crops (Test Acc: 98.35%, Par F1: 0.9016).</div>

      <label for="cropMode">Primary Crop Preprocessing Mode</label>
      <select id="cropMode">
        <option value="raw" selected>raw — Rectangular RGB Crop (4px padding)</option>
        <option value="masked">masked — Elliptical Black-Corner Mask ([0,0,0] corners)</option>
      </select>

      <label for="yoloDevice">Stage 1 YOLOv8n Compute Device</label>
      <select id="yoloDevice">
        <option value="0" selected>GPU: NVIDIA GeForce RTX 2050 (CUDA device=0, auto CPU fallback)</option>
        <option value="cpu">CPU: Force CPU Execution</option>
      </select>
    </div>

    <div class="card">
      <div class="card-title">3. Validated Inference Thresholds</div>
      <div class="grid-2">
        <div>
          <label for="yoloConf">YOLO Conf Thresh</label>
          <input type="number" id="yoloConf" value="0.25" min="0.05" max="0.95" step="0.05">
        </div>
        <div>
          <label for="yoloIou">YOLO NMS IoU</label>
          <input type="number" id="yoloIou" value="0.45" min="0.10" max="0.90" step="0.05">
        </div>
        <div>
          <label for="dedupIou">Deduplication IoU</label>
          <input type="number" id="dedupIou" value="0.50" min="0.10" max="0.95" step="0.05">
        </div>
        <div>
          <label for="cnnThresh">CNN Sigmoid Thresh</label>
          <input type="number" id="cnnThresh" value="0.50" min="0.05" max="0.95" step="0.05">
        </div>
        <div>
          <label for="padPx">Crop Padding (px)</label>
          <input type="number" id="padPx" value="4" min="0" max="16" step="1">
        </div>
        <div>
          <label for="minBoxSize">Min Box Size (px)</label>
          <input type="number" id="minBoxSize" value="16" min="8" max="64" step="2">
        </div>
      </div>

      <label style="margin-top:14px;">Edge-Case Verification Modes (Optional)</label>
      <div class="checkbox-row">
        <input type="checkbox" id="testZeroDet">
        <span>Simulate Zero YOLO Detections (<code>--test-zero-detections</code>)</span>
      </div>
      <div class="checkbox-row">
        <input type="checkbox" id="injectInvalidBoxes">
        <span>Inject Synthetic WBC &amp; Invalid Boxes (<code>--inject-test-invalid-boxes</code>)</span>
      </div>

      <button class="btn-primary" id="runBtn" onclick="runAnalysis()">
        Run Two-Stage Blood-Smear Analysis
      </button>
    </div>
  </section>

  <!-- Right Output Dashboard -->
  <section>
    <div id="errorContainer"></div>

    <div id="welcomeBanner" class="alert-info">
      <strong>Ready for Demonstration:</strong> Upload a full-resolution microscope blood-smear image or select one of the 25 held-out test images on the left, choose your CNN model variant and crop mode, and click <strong>Run Two-Stage Blood-Smear Analysis</strong>.
    </div>

    <div id="resultsContainer" style="display:none;">
      <!-- Separate Count KPIs -->
      <div class="kpi-grid">
        <div class="kpi-box kpi-par">
          <div class="kpi-label">Predicted Parasitized RBCs</div>
          <div class="kpi-val" id="kpiPar">0</div>
          <div class="kpi-sub" id="kpiParSub">Primary CNN Mode</div>
        </div>
        <div class="kpi-box kpi-uninf">
          <div class="kpi-label">Predicted Uninfected RBCs</div>
          <div class="kpi-val" id="kpiUninf">0</div>
          <div class="kpi-sub" id="kpiUninfSub">Valid RBC Candidates</div>
        </div>
        <div class="kpi-box kpi-rate">
          <div class="kpi-label">Experimental Parasitemia</div>
          <div class="kpi-val" id="kpiRate">0.00%</div>
          <div class="kpi-sub" id="kpiRateFrac">0 / 0 RBCs</div>
        </div>
        <div class="kpi-box kpi-wbc">
          <div class="kpi-label">Excluded WBC Detections</div>
          <div class="kpi-val" id="kpiWbc">0</div>
          <div class="kpi-sub">Excluded from RBC denominator</div>
        </div>
        <div class="kpi-box kpi-rej">
          <div class="kpi-label">Rejected / Duplicate Boxes</div>
          <div class="kpi-val" id="kpiRej">0</div>
          <div class="kpi-sub">Clipped, tiny, or IoU &ge; dedup</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">Processing Time</div>
          <div class="kpi-val" id="kpiTime">0.0s</div>
          <div class="kpi-sub" id="kpiTimeSub">Stage 1 + Stage 2</div>
        </div>
      </div>

      <!-- Export & Execution Metadata Card -->
      <div class="card">
        <div class="card-title">
          <span>Active Configuration, Confidence Telemetry &amp; Result Exports</span>
          <span id="runIdBadge" style="font-size:0.78rem; color:var(--muted);"></span>
        </div>
        <div id="execSummaryGrid" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; font-size:0.83rem; margin-bottom:14px;"></div>
        <div id="exportButtons"></div>
      </div>

      <!-- Visual & Tabular Inspection Card -->
      <div class="card">
        <div class="tabs">
          <button class="tab-btn active" onclick="switchTab('tabAnnotated', this)">Annotated Full-Smear Image</button>
          <button class="tab-btn" onclick="switchTab('tabContact', this)">Candidate RBC Contact Sheet</button>
          <button class="tab-btn" onclick="switchTab('tabCells', this)">Per-Cell Predictions &amp; Confidence (<span id="tabCellCount">0</span>)</button>
          <button class="tab-btn" onclick="switchTab('tabExcluded', this)">Excluded WBCs &amp; Rejected Boxes (<span id="tabExclCount">0</span>)</button>
          <button class="tab-btn" onclick="switchTab('tabTxtReport', this)">Formatted Text Report</button>
        </div>

        <div id="tabAnnotated" class="tab-pane active">
          <img id="annotatedImg" class="img-viewer" alt="Annotated Full-Smear Image">
        </div>

        <div id="tabContact" class="tab-pane">
          <img id="contactImg" class="img-viewer" alt="Candidate RBC Crops Contact Sheet">
        </div>

        <div id="tabCells" class="tab-pane">
          <div style="display:flex; gap:12px; margin-bottom:10px; align-items:center;">
            <label style="margin:0;">Filter Class:</label>
            <select id="cellFilter" onchange="renderCellTable()" style="width:220px;">
              <option value="ALL">All Valid RBCs</option>
              <option value="Parasitized">Parasitized RBCs Only</option>
              <option value="Uninfected">Uninfected RBCs Only</option>
            </select>
          </div>
          <div style="max-height:520px; overflow-y:auto;">
            <table>
              <thead>
                <tr>
                  <th>Crop</th>
                  <th>Cell ID</th>
                  <th>Bounding Box [x1,y1,x2,y2]</th>
                  <th>YOLO Conf</th>
                  <th>Primary CNN Class</th>
                  <th>CNN Confidence</th>
                  <th>Raw Crop P(Par)</th>
                  <th>Masked Crop P(Par)</th>
                </tr>
              </thead>
              <tbody id="cellTableBody"></tbody>
            </table>
          </div>
        </div>

        <div id="tabExcluded" class="tab-pane">
          <h4 style="margin-bottom:8px; color:#fbbf24;">Excluded WBC Detections (Class 2)</h4>
          <div id="wbcListContainer" style="margin-bottom:18px;"></div>
          <h4 style="margin-bottom:8px; color:#c084fc;">Rejected / Duplicate Detections</h4>
          <div id="rejectedListContainer"></div>
        </div>

        <div id="tabTxtReport" class="tab-pane">
          <pre class="report-box" id="txtReportBox"></pre>
        </div>
      </div>
    </div>
  </section>
</main>

<script>
let CURRENT_RESULT = null;
let ENV_STATUS = null;

async function loadEnvStatus() {
  try {
    const resp = await fetch('/api/status');
    ENV_STATUS = await resp.json();
    const g = ENV_STATUS.gpu_environment;
    const badges = document.getElementById('hwBadges');
    const gpuClass = g.cuda_available ? 'gpu-ok' : 'cpu-ok';
    const gpuText = g.cuda_available
      ? `Stage 1 GPU: ${g.gpu_name} (${g.free_vram_mb}/${g.total_vram_mb} MB VRAM)`
      : `Stage 1: ${g.gpu_name}`;
    badges.innerHTML = `
      <span class="badge ${gpuClass}">${gpuText}</span>
      <span class="badge cpu-ok">Stage 2: TensorFlow 2.21.0 (venv CPU)</span>
    `;

    const sel = document.getElementById('presetSelect');
    (ENV_STATUS.preset_test_images || []).forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.filename;
      opt.textContent = `${item.filename} [Patient: ${item.patient_id} | GT: ${item.gt_parasitized_rbc}/${item.gt_total_rbc} Par (${item.gt_parasitemia_percent}%)]`;
      sel.appendChild(opt);
    });
  } catch (err) {
    console.error('Status check failed:', err);
  }
}

function onVariantChange() {
  const v = document.getElementById('cnnVariant').value;
  const cropSel = document.getElementById('cropMode');
  const desc = document.getElementById('variantDesc');
  if (v === 'finetuned_raw') {
    cropSel.value = 'raw';
    desc.textContent = 'Fine-tuned on 24,112 native rectangular full-smear crops (Test Acc: 98.35%, Par F1: 0.9016). Recommended mode: raw.';
  } else if (v === 'finetuned_masked') {
    cropSel.value = 'masked';
    desc.textContent = 'Validated on elliptical black-corner masked full-smear crops (Test Acc: 98.10%, Par F1: 0.8905). Recommended mode: masked.';
  } else {
    cropSel.value = 'masked';
    desc.textContent = 'Original single-cell CNN (95.04% NIH test accuracy). Sensitive to unmasked corners; compare raw vs masked!';
  }
}

function switchTab(paneId, btnEl) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(paneId).classList.add('active');
  btnEl.classList.add('active');
}

async function runAnalysis() {
  const errBox = document.getElementById('errorContainer');
  errBox.innerHTML = '';
  const fileInput = document.getElementById('imageUpload');
  const presetVal = document.getElementById('presetSelect').value;

  if (!fileInput.files.length && !presetVal) {
    errBox.innerHTML = `<div class="alert-error"><strong>Input Required:</strong> Please upload a blood-smear image file or choose a preset test image from the dropdown.</div>`;
    return;
  }

  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.textContent = 'Running Stage 1 (YOLOv8n GPU) + Stage 2 (CNN)...';

  try {
    const fd = new FormData();
    if (fileInput.files.length > 0) {
      fd.append('image_file', fileInput.files[0]);
    } else {
      fd.append('preset_filename', presetVal);
    }
    fd.append('cnn_variant', document.getElementById('cnnVariant').value);
    fd.append('crop_mode', document.getElementById('cropMode').value);
    fd.append('yolo_device', document.getElementById('yoloDevice').value);
    fd.append('yolo_conf', document.getElementById('yoloConf').value);
    fd.append('yolo_iou', document.getElementById('yoloIou').value);
    fd.append('dedup_iou', document.getElementById('dedupIou').value);
    fd.append('cnn_thresh', document.getElementById('cnnThresh').value);
    fd.append('pad_px', document.getElementById('padPx').value);
    fd.append('min_box_size', document.getElementById('minBoxSize').value);
    fd.append('test_zero_detections', document.getElementById('testZeroDet').checked ? '1' : '0');
    fd.append('inject_test_invalid_boxes', document.getElementById('injectInvalidBoxes').checked ? '1' : '0');

    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await resp.json();

    if (!data.success) {
      errBox.innerHTML = `<div class="alert-error"><strong>${data.error_type || 'Analysis Error'}:</strong> ${data.error_message || 'Unknown error occurred.'}</div>`;
      return;
    }

    CURRENT_RESULT = data;
    renderResults(data);
  } catch (err) {
    errBox.innerHTML = `<div class="alert-error"><strong>Request Failed:</strong> ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Two-Stage Blood-Smear Analysis';
  }
}

function renderResults(data) {
  document.getElementById('welcomeBanner').style.display = 'none';
  document.getElementById('resultsContainer').style.display = 'block';

  const s = data.summary;
  const cnt = s.counts;
  const parEst = s.parasitemia_estimate;
  const times = s.processing_time_seconds;
  const conf = s.confidence_statistics || {};
  const hw = s.hardware_execution || {};

  document.getElementById('kpiPar').textContent = cnt.cnn_parasitized_rbc_count;
  document.getElementById('kpiParSub').textContent = `Mode: ${s.models.primary_crop_preprocessing_mode} | Variant: ${s.models.stage2_cnn_variant}`;
  document.getElementById('kpiUninf').textContent = cnt.cnn_uninfected_rbc_count;
  document.getElementById('kpiUninfSub').textContent = `Out of ${cnt.valid_candidate_rbc_crops} valid RBC crops`;
  document.getElementById('kpiRate').textContent = `${parEst.estimated_parasitemia_percent.toFixed(2)}%`;
  document.getElementById('kpiRateFrac').textContent = parEst.calculation_fraction;
  document.getElementById('kpiWbc').textContent = cnt.excluded_wbc_detections;
  document.getElementById('kpiRej').textContent = cnt.rejected_or_duplicate_boxes;
  document.getElementById('kpiTime').textContent = `${times.total_pipeline_seconds.toFixed(2)}s`;
  document.getElementById('kpiTimeSub').textContent = `S1: ${times.stage1_yolo_detection_and_cropping.toFixed(2)}s | S2: ${times.stage2_cnn_classification.toFixed(2)}s`;

  document.getElementById('runIdBadge').textContent = `Run ID: ${data.run_id}`;

  const yStat = conf.yolo_detection_confidence || {mean:0, median:0, min:0, max:0};
  const cStat = conf.cnn_overall_confidence || {mean:0, median:0, min:0, max:0};
  const pStat = conf.cnn_parasitized_confidence || {mean:0, median:0, min:0, max:0};

  let gtHtml = '<em>Uploaded image (no ground-truth file in yolo_malaria/labels)</em>';
  if (data.ground_truth_reference && data.ground_truth_reference.has_ground_truth) {
    const gt = data.ground_truth_reference;
    gtHtml = `Split: <strong>${gt.split}</strong> (Patient <code>${gt.patient_id}</code>) | GT Parasitized: <strong>${gt.gt_parasitized_rbc} / ${gt.gt_total_rbc} (${gt.gt_parasitemia_percent}%)</strong> | GT WBC: ${gt.gt_wbc}`;
  }

  document.getElementById('execSummaryGrid').innerHTML = `
    <div>
      <div><strong>Image:</strong> <code>${s.image_filename}</code> (${s.image_dimensions.width}&times;${s.image_dimensions.height} px)</div>
      <div><strong>Stage 1 Hardware:</strong> ${hw.stage1_device} (Peak VRAM: ${hw.stage1_peak_vram_mb} MB)</div>
      <div><strong>Stage 2 Hardware:</strong> ${hw.stage2_device}</div>
      ${hw.gpu_fallback_triggered ? `<div style="color:#fbbf24;"><strong>GPU Fallback Notice:</strong> ${hw.gpu_fallback_reason}</div>` : ''}
    </div>
    <div>
      <div><strong>Selected CNN Variant:</strong> <code>${s.models.stage2_cnn_variant}</code></div>
      <div><strong>CNN Checkpoint:</strong> <code>${s.models.stage2_cnn_checkpoint.split('\\').pop()}</code></div>
      <div><strong>Primary Crop Mode:</strong> <code>${s.models.primary_crop_preprocessing_mode}</code> (Thresh: ${s.thresholds_and_parameters.cnn_parasitized_threshold})</div>
      <div><strong>Audit Comparison:</strong> Raw=${parEst.raw_crop_parasitemia_percent.toFixed(2)}% (${cnt.cnn_raw_crop_parasitized_count} Par) vs Masked=${parEst.ellipse_masked_parasitemia_percent.toFixed(2)}% (${cnt.cnn_ellipse_masked_parasitized_count} Par)</div>
    </div>
    <div>
      <div><strong>YOLO Detection Conf (Mean / Min–Max):</strong> ${(yStat.mean*100).toFixed(1)}% (${(yStat.min*100).toFixed(1)}%–${(yStat.max*100).toFixed(1)}%)</div>
      <div><strong>CNN Overall Conf (Mean / Median):</strong> ${(cStat.mean*100).toFixed(1)}% / ${(cStat.median*100).toFixed(1)}%</div>
      <div><strong>CNN Parasitized Conf (Mean):</strong> ${pStat.count > 0 ? (pStat.mean*100).toFixed(1)+'%' : 'N/A (0 Par)'}</div>
      <div><strong>Ground-Truth Reference:</strong> ${gtHtml}</div>
    </div>
  `;

  document.getElementById('exportButtons').innerHTML = `
    <a class="btn-export" href="${data.urls.export_csv}" download>Download Per-Cell CSV (cell_predictions_${data.stem}.csv)</a>
    <a class="btn-export" href="${data.urls.export_summary_json}" download>Download Summary JSON (field_summary_${data.stem}.json)</a>
    <a class="btn-export" href="${data.urls.export_rejected_json}" download>Download Excluded/Rejected JSON</a>
    <a class="btn-export" href="${data.urls.export_txt}" download>Download Text Report (.txt)</a>
    ${data.urls.annotated_image ? `<a class="btn-export" href="${data.urls.annotated_image}&download=1" download>Download Annotated Image (.jpg)</a>` : ''}
  `;

  const annEl = document.getElementById('annotatedImg');
  if (data.urls.annotated_image) {
    annEl.src = data.urls.annotated_image + '&t=' + Date.now();
    annEl.style.display = 'block';
  } else {
    annEl.style.display = 'none';
  }

  const csEl = document.getElementById('contactImg');
  if (data.urls.contact_sheet) {
    csEl.src = data.urls.contact_sheet + '&t=' + Date.now();
    csEl.style.display = 'block';
  } else {
    csEl.style.display = 'none';
  }

  document.getElementById('tabCellCount').textContent = data.cell_predictions.length;
  document.getElementById('tabExclCount').textContent =
    (data.rejected_and_excluded.excluded_wbc_count || 0) + (data.rejected_and_excluded.rejected_or_duplicate_count || 0);
  document.getElementById('txtReportBox').textContent = data.text_report || '';

  renderCellTable();
  renderExcludedAndRejected(data.rejected_and_excluded);
}

function renderCellTable() {
  if (!CURRENT_RESULT) return;
  const filterVal = document.getElementById('cellFilter').value;
  const rows = CURRENT_RESULT.cell_predictions || [];
  const filtered = rows.filter(r => filterVal === 'ALL' || r.cnn_predicted_class === filterVal);

  const tbody = document.getElementById('cellTableBody');
  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:var(--muted);">No candidate RBC cells match the current filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(r => {
    const isPar = r.cnn_predicted_class === 'Parasitized';
    const pillClass = isPar ? 'pill-par' : 'pill-uninf';
    const cropUrl = `/api/file?run_id=${CURRENT_RESULT.run_id}&file=${encodeURIComponent(r.crop_rel_path.replace(/\\/g, '/'))}`;
    return `
      <tr>
        <td><img class="crop-thumb" src="${cropUrl}" loading="lazy" alt="${r.crop_id}"></td>
        <td><code>${r.crop_id}</code></td>
        <td>[${r.bbox_x1}, ${r.bbox_y1}, ${r.bbox_x2}, ${r.bbox_y2}] (${r.bbox_width}&times;${r.bbox_height}px)</td>
        <td>${(parseFloat(r.yolo_confidence)*100).toFixed(1)}%</td>
        <td><span class="pill ${pillClass}">${r.cnn_predicted_class}</span></td>
        <td><strong>${(parseFloat(r.cnn_confidence)*100).toFixed(2)}%</strong></td>
        <td>${(parseFloat(r.cnn_raw_parasitized_prob)*100).toFixed(2)}% (${r.cnn_raw_predicted_class})</td>
        <td>${(parseFloat(r.cnn_masked_parasitized_prob)*100).toFixed(2)}% (${r.cnn_masked_predicted_class})</td>
      </tr>
    `;
  }).join('');
}

function renderExcludedAndRejected(rejData) {
  const wbcs = rejData.excluded_wbcs || [];
  const rejs = rejData.rejected_boxes || [];

  const wbcBox = document.getElementById('wbcListContainer');
  if (!wbcs.length) {
    wbcBox.innerHTML = `<div class="hint">0 WBC detections excluded in this microscope field.</div>`;
  } else {
    wbcBox.innerHTML = `<table><thead><tr><th>Class</th><th>YOLO Confidence</th><th>Clipped BBox</th><th>Exclusion Reason</th></tr></thead><tbody>` +
      wbcs.map(w => `<tr><td>${w.yolo_class_name} (ID ${w.yolo_class_id})</td><td>${(w.yolo_confidence*100).toFixed(1)}%</td><td>[${w.clipped_bbox.join(', ')}]</td><td>${w.reason}</td></tr>`).join('') +
      `</tbody></table>`;
  }

  const rejBox = document.getElementById('rejectedListContainer');
  if (!rejs.length) {
    rejBox.innerHTML = `<div class="hint">0 rejected or duplicate detections in this microscope field.</div>`;
  } else {
    rejBox.innerHTML = `<table><thead><tr><th>Rejection / Suppression Reason</th><th>Details</th></tr></thead><tbody>` +
      rejs.map(r => `<tr><td>${r.reason}</td><td><code>${JSON.stringify(r.clipped_bbox || (r.suppressed_detection && r.suppressed_detection.bbox) || {})}</code></td></tr>`).join('') +
      `</tbody></table>`;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  loadEnvStatus();
  document.getElementById('imageUpload').addEventListener('change', () => {
    if (document.getElementById('imageUpload').files.length > 0) {
      document.getElementById('presetSelect').value = '';
    }
  });
});
</script>
</body>
</html>
"""


def parse_multipart_form_data(headers, body_bytes):
    """
    Lightweight, zero-dependency multipart/form-data parser compatible with Python 3.11+.
    Returns (fields_dict, files_dict).
    """
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        return {}, {}

    boundary = content_type.split("boundary=")[1].split(";")[0].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    delimiter = ("--" + boundary).encode("utf-8")
    parts = body_bytes.split(delimiter)

    fields = {}
    files = {}

    for part in parts[1:]:
        if part.startswith(b"--") or len(part.strip()) == 0:
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]

        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue

        raw_headers = part[:header_end].decode("utf-8", errors="replace")
        content = part[header_end + 4:]

        disp_name = None
        filename = None
        for hline in raw_headers.split("\r\n"):
            if hline.lower().startswith("content-disposition:"):
                for token in hline.split(";"):
                    token = token.strip()
                    if token.startswith("name="):
                        disp_name = token.split("=", 1)[1].strip('"')
                    elif token.startswith("filename="):
                        filename = token.split("=", 1)[1].strip('"')

        if not disp_name:
            continue

        if filename is not None and filename != "":
            files[disp_name] = {
                "filename": os.path.basename(filename),
                "content": content
            }
        else:
            fields[disp_name] = content.decode("utf-8", errors="replace")

    return fields, files


class SmartCellRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write(f"[SmartCell WebApp] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, status_code, payload):
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            raw = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/status":
            refresh = qs.get("refresh", ["0"])[0] == "1"
            status = query_environment_status(force_refresh=refresh)
            self._send_json(200, status)
            return

        if path == "/api/file":
            run_id = qs.get("run_id", [""])[0]
            rel_file = qs.get("file", [""])[0]
            download = qs.get("download", ["0"])[0] == "1"

            # Prevent path traversal outside DEMO_RUNS_ROOT
            safe_run = os.path.basename(run_id)
            run_dir = os.path.abspath(os.path.join(DEMO_RUNS_ROOT, safe_run))
            target_path = os.path.abspath(os.path.join(run_dir, rel_file))

            if not target_path.startswith(run_dir) or not os.path.isfile(target_path):
                self._send_json(404, {"error": "Requested artifact file not found."})
                return

            ctype, _ = mimetypes.guess_type(target_path)
            if not ctype:
                ctype = "application/octet-stream"

            with open(target_path, "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if download:
                fname = os.path.basename(target_path)
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json(404, {"error": f"Route '{path}' not found."})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send_json(404, {"error": "Unknown POST endpoint."})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            body_bytes = self.rfile.read(content_len)
            fields, files = parse_multipart_form_data(self.headers, body_bytes)

            os.makedirs(UPLOADS_DIR, exist_ok=True)
            image_path = None

            if "image_file" in files and len(files["image_file"]["content"]) > 0:
                orig_fn = os.path.basename(files["image_file"]["filename"])
                upload_sub = os.path.join(UPLOADS_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
                os.makedirs(upload_sub, exist_ok=True)
                upload_path = os.path.join(upload_sub, orig_fn)
                with open(upload_path, "wb") as uf:
                    uf.write(files["image_file"]["content"])
                image_path = upload_path
            elif fields.get("preset_filename"):
                preset_fn = os.path.basename(fields["preset_filename"])
                cand_path = os.path.join(TEST_IMAGES_DIR, preset_fn)
                if not os.path.exists(cand_path):
                    self._send_json(400, {
                        "success": False,
                        "error_type": "PresetNotFoundError",
                        "error_message": f"Preset test image '{preset_fn}' not found."
                    })
                    return
                image_path = cand_path
            else:
                self._send_json(400, {
                    "success": False,
                    "error_type": "MissingImageError",
                    "error_message": "No image file uploaded or preset selected."
                })
                return

            result = execute_pipeline_run(
                image_path=image_path,
                cnn_variant=fields.get("cnn_variant", "finetuned_raw"),
                crop_mode=fields.get("crop_mode", "raw"),
                yolo_device=fields.get("yolo_device", "0"),
                yolo_conf=fields.get("yolo_conf", 0.25),
                yolo_iou=fields.get("yolo_iou", 0.45),
                dedup_iou=fields.get("dedup_iou", 0.50),
                cnn_thresh=fields.get("cnn_thresh", 0.50),
                pad_px=fields.get("pad_px", 4),
                min_box_size=fields.get("min_box_size", 16),
                test_zero_detections=(fields.get("test_zero_detections") == "1"),
                inject_test_invalid_boxes=(fields.get("inject_test_invalid_boxes") == "1")
            )
            status_code = 200 if result.get("success") else 400
            self._send_json(status_code, result)
        except Exception as e:
            self._send_json(500, {
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            })


def main():
    parser = argparse.ArgumentParser(description="SmartCell Phase 7 Interactive Demonstration Web Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve the web application (default: 8501)")
    args = parser.parse_args()

    os.makedirs(DEMO_RUNS_ROOT, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    print("=" * 78)
    print("SMARTCELL PHASE 7: INTERACTIVE LOCAL DEMONSTRATION APPLICATION")
    print("=" * 78)
    print("DISCLAIMER: Experimental academic research prototype. Not for clinical use.")
    status = query_environment_status(force_refresh=True)
    g = status["gpu_environment"]
    print(f"Stage 1 GPU Env:  {g['gpu_name']} (CUDA Available: {g['cuda_available']}, PyTorch: {g['pytorch_version']})")
    print(f"Stage 2 CNN Env:  {VENV_TF_PYTHON}")
    print(f"Output Root Dir:  {DEMO_RUNS_ROOT}")
    print(f"Web UI URL:       http://{args.host}:{args.port}")
    print("=" * 78)

    server = ThreadingHTTPServer((args.host, args.port), SmartCellRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SmartCell demonstration server.")
        server.server_close()


if __name__ == "__main__":
    main()
