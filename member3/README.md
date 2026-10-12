# SmartCell — Member 3 Weekly Project Work: Two-Stage YOLOv8n + Domain-Adapted CNN Malaria Blood-Smear Screening System

> **IMPORTANT RESEARCH PROTOTYPE DISCLAIMER**  
> **SmartCell** is an experimental academic research prototype developed for a college engineering project. It is **not** a clinically validated medical diagnostic device, has **not** undergone regulatory clearance, and must **never** be used for clinical diagnosis, patient screening, or treatment decisions. All outputs represent experimental AI screening estimates (`predicted classes`, `confidence scores`, and `experimental image-level parasitemia estimates`).

---

## 1. Weekly Work Summary

This directory (`member3/`) contains the complete implementation, experiments, evaluation reports, visual artifacts, and interactive demonstration application completed by **Member 3** for the **SmartCell** AI-based blood-cell screening project (`D:\smartcell`).

### Completed Milestones Included in `member3/`
1. **CNN-Based Malaria Single-Cell Classification & Transfer Learning Comparison (Phases 1–2)**:
   - Verified `27,558` NIH single-cell erythrocyte images (`0` corrupted, `0` MD5 duplicates) and created a balanced `70% / 15% / 15%` split (`19,290` train, `4,132` val, `4,136` test; seed `42`).
   - Trained a custom 4-block Convolutional Neural Network (`1,848,001` parameters) achieving **`95.04%` test accuracy**, **`97.12%` precision**, **`92.84%` recall**, and **`0.9493` F1** on the `4,136` held-out single-cell test set, outperforming `MobileNetV2` transfer learning (`92.07%` accuracy, `0.9155` F1).
2. **YOLOv8n Full-Smear Blood-Cell Detection & Localization Baseline (Phases 3–4)**:
   - Converted the official **NIH-NLM Thin Blood Smears *Plasmodium falciparum*** (`Polygon Set`) dataset (`165` full-resolution $5312 \times 2988\text{ px}$ microscope fields across `33` patients; `34,264` polygonal cell annotations) into normalized YOLO format (`0` invalid or degenerate boxes).
   - Enforced a strict **patient-stratified split (`23` train / `5` val / `5` test patients; `0` patient leakage)** and trained `YOLOv8n` for `50` epochs, achieving **`97.72%` validation $\text{mAP@50}$** and **`96.44%` test $\text{mAP@50}$** on `Uninfected RBC`, with **`98.25%` overall ground-truth cell localization recall** on the 25 held-out test images.
3. **Two-Stage YOLO + CNN Detection & Classification Pipeline (Phase 5)**:
   - Engineered an automated two-stage orchestrator (`scripts/run_smartcell_pipeline.py`) that uses `YOLOv8n` (`CUDA device=0` on NVIDIA GeForce RTX 2050 GPU, with automatic CPU fallback) as a class-agnostic erythrocyte localizer (excluding `class 2 [WBC]` detections and deduplicating overlapping boxes at $\text{IoU} \ge 0.50$) and passes native-resolution RGB cell crops ($+4\text{ px}$ context padding) to the dedicated `CNN` classifier.
   - Identified the **background domain-shift vulnerability**: the Phase 1 CNN trained on black-background (`[0, 0, 0]`) segmented cells produced elevated false positives on raw rectangular smear crops (`55.38%` predicted parasitemia vs. `34.47%` ground truth on test smear `IMG_20150819_133447.jpg`) due to bright pink plasma corners and neighboring cell borders.
4. **Raw & Elliptical-Masked Crop Domain Adaptation Experiments (Phase 6)**:
   - Extracted `34,208` patient-split ground-truth RBC crops (`24,112` train, `5,302` val, `4,794` test; `0` WBCs, `0` patient overlap) in both **Raw Rectangular (`+4 px` padding)** and **Elliptical-Masked (`[0, 0, 0]` black corners)** formats (`scripts/prepare_domain_crops.py`).
   - Fine-tuned the baseline CNN with class-balanced weighting (`scripts/train_eval_domain_adaptation.py`), producing `malaria_cnn_finetuned_raw_best.keras` and `malaria_cnn_finetuned_masked_best.keras`.
   - On the `4,794` held-out ground-truth test crops (5 patients), **`Finetuned_CNN_Raw`** improved test accuracy from `91.80%` $\rightarrow$ **`98.35%`**, specificity from `91.13%` $\rightarrow$ **`98.53%`**, and Parasitized F1 from `0.6562` $\rightarrow$ **`0.9016`**, reducing false positives by **`83.4%`** (`392` $\rightarrow$ `65`) without requiring artificial corner masking.
5. **End-to-End Two-Stage Evaluation & Parasitemia Estimation (Phase 6)**:
   - Evaluated all 4 two-stage strategies across all `25` held-out test blood-smear images (`4,794` GT RBCs) while strictly penalizing Stage 1 detection misses (`14` missed Parasitized RBCs counted directly as end-to-end false negatives `FN_Miss`) (`scripts/evaluate_two_stage_test_set.py`).
   - **`YOLOv8n` + `Finetuned_CNN_Raw`** achieved the best end-to-end performance: **`0.8698` End-to-End Parasitized F1**, **`82.89%` precision**, **`91.49%` sensitivity (including Stage 1 misses)**, and **`1.12%` Mean Absolute Error (MAE)** on image-level and patient-level parasitemia estimation (down from `7.36%` MAE with the unadapted baseline).
6. **Interactive SmartCell Web Demonstration Application (Phase 7)**:
   - Built a self-contained local web application (`scripts/smartcell_demo_app.py` and `launch_smartcell_demo.bat`) on Python's standard `ThreadingHTTPServer` (`http://127.0.0.1:8501`) with zero external web-framework dependencies.
   - Supports full-smear image uploads, preset held-out test image selection, live switching across all `3` CNN checkpoints (`finetuned_raw`, `finetuned_masked`, `original`) and `2` crop modes (`raw`, `masked`), configurable YOLO/CNN thresholds, separate KPI cards for Parasitized RBCs, Uninfected RBCs, Excluded WBCs, and Rejected/Duplicate boxes, confidence telemetry, annotated full-field and contact-sheet inspection tabs, and one-click CSV/JSON/JPG exports.
7. **Phase 7 & Phase 8 Comprehensive Verification, Reproducibility & Security Audits**:
   - Executed automated end-to-end verification suites (`scripts/verify_phase7_demo.py` and `scripts/verify_phase8_final.py`) confirming `100%` numerical consistency across the browser UI, JSON summaries, and per-cell CSV exports, byte-for-byte deterministic reproducibility across repeated runs (`max_per_cell_probability_difference = 0.0`), safe handling of corrupted uploads (`HTTP 400`), zero-detection safe denominator (`0 / 0`), path-traversal protection (`HTTP 404`), and `0` orphaned worker subprocesses.

---

## 2. Directory Structure (`member3/`)

```text
member3/
├── README.md                          # Summary of Member 3 weekly work, methodology, results & setup
├── requirements.txt                   # Core Python dependencies for TensorFlow/CNN & evaluation
├── launch_smartcell_demo.bat          # One-click Windows launcher for the SmartCell web demo app
├── src/                               # Primary application & two-stage pipeline source modules
│   ├── smartcell_demo_app.py          # Interactive local web application server (HTTP 127.0.0.1:8501)
│   ├── run_smartcell_pipeline.py      # Two-stage YOLOv8n (GPU/CPU) -> CNN (CPU) pipeline orchestrator
│   └── predict_malaria.py             # Single-cell CNN inference utility
├── scripts/                           # Complete data prep, training, evaluation & verification scripts
│   ├── check_dataset.py               # Corrupted image integrity verifier (27,558 single-cell images)
│   ├── check_duplicates.py            # MD5 hash exact duplicate scanner
│   ├── split_dataset.py               # Stratified 70/15/15 single-cell dataset splitter (seed 42)
│   ├── train_cnn.py                   # Phase 1 custom 4-block CNN training script
│   ├── evaluate_cnn.py                # Phase 1 CNN untouched test-set evaluation
│   ├── error_analysis.py              # False positive / false negative error analysis & crop exporter
│   ├── train_mobilenetv2.py           # Phase 2 MobileNetV2 transfer learning training script
│   ├── evaluate_mobilenetv2.py        # Phase 2 MobileNetV2 test-set evaluation
│   ├── compare_models.py              # Phase 2 Baseline CNN vs. MobileNetV2 benchmark comparison
│   ├── predict_malaria.py             # Single-image CNN prediction CLI
│   ├── test_model.py                  # Random test-set sample verification script
│   ├── prepare_yolo_malaria.py        # Phase 3 Polygon Set -> YOLO converter & patient-stratified splitter
│   ├── verify_yolo_labels.py          # Phase 3 YOLO bounding-box verifier & visual preview generator
│   ├── preflight_check.py             # Phase 4 hardware/dataset preflight verification
│   ├── smoke_test_yolo.py             # Phase 4 1-epoch YOLOv8n pipeline smoke test
│   ├── train_yolo_malaria.py          # Phase 4 50-epoch YOLOv8n blood-cell detection training
│   ├── evaluate_yolo_malaria.py       # Phase 4 YOLOv8n val/test evaluation & GT cell counting analysis
│   ├── run_smartcell_pipeline.py      # Phase 5–7 two-stage YOLOv8n + CNN blood-smear pipeline CLI
│   ├── prepare_domain_crops.py        # Phase 6 patient-split raw & elliptical-masked GT RBC crop generator
│   ├── train_eval_domain_adaptation.py# Phase 6 domain-adapted CNN fine-tuning & 4-strategy evaluation
│   ├── evaluate_two_stage_test_set.py # Phase 6 end-to-end 25-image two-stage evaluation (penalizing misses)
│   ├── smartcell_demo_app.py          # Phase 7 interactive local demonstration web application
│   ├── verify_phase7_demo.py          # Phase 7 automated HTTP & pipeline verification suite
│   └── verify_phase8_final.py         # Phase 8 reproducibility, 3x2 matrix & security verification suite
├── docs/                              # Project & live demonstration documentation
│   ├── DEMO_GUIDE.md                  # 5–7 minute step-by-step college project live demonstration guide
│   └── FULL_PROJECT_LOG.md            # Complete Phases 1–8 engineering log from D:\smartcell\README.md
├── models/                            # Class mapping & documentation for local model checkpoints
│   ├── malaria_class_names.json       # Binary class index mapping ({"0": "Uninfected", "1": "Parasitized"})
│   └── README.md                      # Inventory, byte sizes & setup guide for excluded .keras / .pt weights
├── dataset/                           # Dataset configuration, split metadata & normalized YOLO labels
│   ├── README.md                      # Split statistics & documentation for single-cell, YOLO & crop datasets
│   ├── data.yaml                      # YOLO dataset configuration file
│   ├── Dataset_statistics.xlsx        # Official NIH-NLM Thin Blood Smears patient & annotation statistics
│   ├── yolo_preparation_report.json   # Machine-readable patient-stratified split & annotation report
│   ├── yolo_preparation_report.txt    # Formatted summary of the 33-patient (23/5/5) YOLO split
│   └── yolo_malaria/
│       ├── data.yaml                  # YOLO config relative to yolo_malaria/
│       └── labels/                    # All 165 normalized YOLO label .txt files (34,264 cell boxes)
│           ├── train/                 # 115 training image label files (23 patients, 24,154 cells)
│           ├── val/                   # 25 validation image label files (5 patients, 5,310 cells)
│           └── test/                  # 25 held-out test image label files (5 patients, 4,800 cells)
└── results/                           # Evaluation metrics, verification reports, plots & visual outputs
    ├── metrics/                       # All Phase 1–8 JSON, CSV, and TXT evaluation & verification reports
    ├── demo_app_runs/                 # Phase 7 & Phase 8 verification reports and 13 verified demo run folders
    ├── domain_adaptation/             # Phase 6 domain adaptation reports, per-patient CSVs & training/CM plots
    ├── smartcell_pipeline/            # Phase 5 & Phase 6 single-image, batch & edge-case pipeline outputs
    ├── yolo_malaria_run/              # Phase 4 YOLOv8n training curves, PR/F1 curves, val/test eval & predictions
    ├── yolo_preview/                  # Phase 3 verified ground-truth bounding-box overview & crop previews
    ├── confusion_matrix/              # Phase 1–2 Baseline CNN & MobileNetV2 confusion matrices
    └── graphs/                        # Phase 1–2 Baseline CNN & MobileNetV2 accuracy and loss curves
```

---

## 3. Environment Setup & Execution Instructions

### Dual-Environment Architecture (Native Windows)
Because `TensorFlow >= 2.11` (`TensorFlow 2.21.0`) executes on CPU on native Windows while `PyTorch 2.14.1+cu130` (`Ultralytics 8.4.37`) natively supports the laptop's **NVIDIA GeForce RTX 2050 GPU (`CUDA device=0`, 4 GB VRAM)**, the local workspace (`D:\smartcell`) uses two isolated virtual environments bridged automatically by `run_smartcell_pipeline.py` and `smartcell_demo_app.py`:

1. **Stage 1 GPU Detection Environment (`D:\smartcell\venv_yolo_gpu`)**:
   - Python 3.13, `torch==2.14.1+cu130`, `torchvision`, `ultralytics==8.4.37` (automatically falls back to `D:\smartcell\venv_yolo` CPU if CUDA is unavailable).
2. **Stage 2 CNN & Web App Environment (`D:\smartcell\venv`)**:
   - Python 3.11.9, `tensorflow==2.21.0`, `keras==3.15.1`, `opencv-python>=4.9.0`, `pillow>=10.0.0`, `numpy>=1.26.0`, `scikit-learn>=1.4.0`, `matplotlib>=3.8.0` (see [`requirements.txt`](requirements.txt)).

### Running the Interactive SmartCell Demo Application
```powershell
# Option 1: One-click Windows batch launcher
D:\smartcell\launch_smartcell_demo.bat

# Option 2: Direct CLI launch from D:\smartcell
D:\smartcell\venv\Scripts\python.exe D:\smartcell\scripts\smartcell_demo_app.py --host 127.0.0.1 --port 8501
```
Then open **`http://127.0.0.1:8501`** in your browser. See [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) for the complete 5–7 minute demonstration sequence.

### Running the Two-Stage Pipeline via CLI
```powershell
D:\smartcell\venv\Scripts\python.exe D:\smartcell\scripts\run_smartcell_pipeline.py `
  --image D:\smartcell\yolo_malaria\images\test\IMG_20150819_133447.jpg `
  --output-dir D:\smartcell\results\smartcell_pipeline\phase6_finetuned_raw_test_IMG_20150819_133447 `
  --cnn-variant finetuned_raw `
  --crop-mode raw `
  --yolo-device 0
```

---

## 4. Methodology & Key Experimental Results

### 4.1 Single-Cell Malaria Classification (Phases 1–2)
Evaluated on `4,136` untouched single-cell test images (`2,068` Parasitized, `2,068` Uninfected) from `D:\smartcell\dataset\test` (reports in [`results/metrics/model_comparison.txt`](results/metrics/model_comparison.txt)):

| Model | Parameters | Test Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom 4-Block CNN (`malaria_cnn_best.keras`)** | `1,848,001` | **`95.04%`** | `97.12%` | **`92.84%`** | `97.24%` | **`0.9493`** | **`0.9873`** |
| **MobileNetV2 Transfer Learning (`malaria_mobilenetv2_best.keras`)** | `2,422,209` | `92.07%` | **`97.96%`** | `85.93%` | **`98.21%`** | `0.9155` | `0.9803` |

### 4.2 YOLOv8n Blood-Cell Detection Baseline (Phases 3–4)
Trained for `50` epochs on `115` full-smear images (`23` patients) and evaluated on `25` validation images (`5` patients) and `25` test images (`5` held-out patients) (report in [`results/metrics/yolo_malaria_training_report.json`](results/metrics/yolo_malaria_training_report.json)):

- **Uninfected RBC (`class 0`) Detection**: **`97.72%` Validation $\text{mAP@50}$** (`79.56%` $\text{mAP@50:95}$) and **`96.44%` Test $\text{mAP@50}$** (`79.52%` $\text{mAP@50:95}$).
- **Parasitized RBC (`class 1`) & WBC (`class 2`) Single-Stage Detection**: Due to severe class imbalance (`33,071` Uninfected vs. `1,142` Parasitized RBCs [`29.6 : 1`] and only `37` training WBCs), single-stage YOLOv8n achieved only `2.07%` test $\text{mAP@50}$ on `Parasitized RBC` and `0.00%` on `WBC`.
- **Crucial Finding (Class-Agnostic Localization)**: Despite misclassifying infected cells as `class 0 (Uninfected RBC)`, YOLOv8n localized **`98.25%` (`4,716 / 4,800`)** of all ground-truth cells on the test set (`97.93%` [`4,695 / 4,794`] of all ground-truth RBCs at greedy 1-to-1 matching), motivating our **Two-Stage Architecture** where Stage 1 localizes candidate RBCs and Stage 2 classifies infection status.

### 4.3 Phase 6 Domain Adaptation on Ground-Truth Full-Smear RBC Crops
Evaluated across **`4,794` held-out test RBC crops** (`4,418` Uninfected, `376` Parasitized from the `5` test patients; report in [`results/metrics/phase6_model_comparison_report.json`](results/metrics/phase6_model_comparison_report.json)):

| Strategy | Model Checkpoint | Crop Mode | Test Accuracy | Parasitized Precision | Parasitized Sensitivity | Uninfected Specificity | Parasitized F1 | Confusion Matrix (`TP / FP / TN / FN`) | 5-Patient Parasitemia MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: `Orig_CNN_Raw`** | `malaria_cnn_best.keras` | `raw` | `91.80%` | `48.96%` | **`99.73%`** | `91.13%` | `0.6562` | `375 / 392 / 4026 / 1` | `8.26%` |
| **B: `Orig_CNN_Masked`** | `malaria_cnn_best.keras` | `masked` | `98.10%` | `81.32%` | `98.40%` | `98.08%` | `0.8905` | `370 / 85 / 4333 / 6` | `1.54%` |
| **C: `Finetuned_CNN_Raw`** *(Recommended)* | `malaria_cnn_finetuned_raw_best.keras` | `raw` | **`98.35%`** | **`84.78%`** | `96.28%` | **`98.53%`** | **`0.9016`** | **`362 / 65 / 4353 / 14`** | **`1.11%`** |
| **D: `Finetuned_CNN_Masked`** | `malaria_cnn_finetuned_masked_best.keras` | `masked` | `98.10%` | `81.32%` | `98.40%` | `98.08%` | `0.8905` | `370 / 85 / 4333 / 6` | `1.54%` |

### 4.4 End-to-End Two-Stage Evaluation Across All 25 Held-Out Test Smears (Penalizing Stage 1 Misses)
Executed via [`scripts/evaluate_two_stage_test_set.py`](scripts/evaluate_two_stage_test_set.py) across all `25` test images (`4,794` GT RBCs). Stage 1 `YOLOv8n` (`CUDA device=0`, `Peak VRAM: 29.61 MB`) matched `4,695 / 4,794` GT RBCs (`362 / 376` Parasitized matched; **`14` Parasitized RBCs missed by Stage 1 counted strictly as End-to-End False Negatives `FN_Miss`**; report in [`results/metrics/phase6_two_stage_end_to_end_report.json`](results/metrics/phase6_two_stage_end_to_end_report.json)):

| Two-Stage Pipeline Strategy | End-to-End Parasitized Precision | End-to-End Sensitivity (incl. 14 Stage 1 Misses) | End-to-End Parasitized F1 | Matched RBC Accuracy | End-to-End Counts (`TP / FP_Tot / TN / FN_CNN / FN_Miss / FN_Tot`) | 25-Image Parasitemia MAE | 5-Patient Parasitemia MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: `YOLOv8n` + `Orig_CNN_Raw`** | `50.42%` | **`96.28%`** (`362/376`) | `0.6618` | `92.89%` | `362 / 356 / 3998 / 0 / 14 / 14` | `7.36%` | `7.30%` |
| **B: `YOLOv8n` + `Orig_CNN_Masked`** | `78.54%` | `94.41%` (`355/376`) | `0.8575` | `98.10%` | `355 / 97 / 4251 / 7 / 14 / 21` | `1.81%` | `1.77%` |
| **C: `YOLOv8n` + `Finetuned_CNN_Raw`** *(Recommended)* | **`82.89%`** | `91.49%` (`344/376`) | **`0.8698`** | **`98.36%`** | **`344 / 71 / 4274 / 18 / 14 / 32`** | **`1.12%`** | **`1.12%`** |
| **D: `YOLOv8n` + `Finetuned_CNN_Masked`** | `78.54%` | `94.41%` (`355/376`) | `0.8575` | `98.10%` | `355 / 97 / 4251 / 7 / 14 / 21` | `1.81%` | `1.77%` |

### 4.5 Phase 7 & Phase 8 Verification Summary
Full verification reports are stored in [`results/metrics/phase7_demo_verification_report.json`](results/metrics/phase7_demo_verification_report.json), [`results/metrics/phase8_final_verification_report.json`](results/metrics/phase8_final_verification_report.json), [`results/demo_app_runs/phase7_demo_verification_report.txt`](results/demo_app_runs/phase7_demo_verification_report.txt), and [`results/demo_app_runs/phase8_final_verification_report.txt`](results/demo_app_runs/phase8_final_verification_report.txt):
- **High-Parasitemia Held-Out Smear (`IMG_20150819_133447.jpg`, GT `71 / 206 = 34.47%`)**: Detected `195` valid RBCs, predicted **`74` Parasitized (`37.95%`)** and `121` Uninfected in `~16.1–17.2s` (`Peak GPU VRAM: 23.73 MB`).
- **Low-Parasitemia Held-Out Smear (`IMG_20151203_152841.jpg`, GT `1 / 165 = 0.61%`)**: Detected `169` valid RBCs, predicted **`1` Parasitized (`0.59%`)** and `168` Uninfected.
- **Uninfected Control Smear (`IMG_20151207_114038.jpg`, GT `0 / 208 = 0.00%`)**: Detected `210` valid RBCs, predicted **`1` Parasitized (`0.48%`)** and `209` Uninfected.
- **Deterministic Reproducibility**: Two consecutive runs on `IMG_20150819_133447.jpg` produced `100%` byte-for-byte identical CSV exports (`SHA-256: 9d3e3fd6fc207291...`), annotated full-smear JPGs (`SHA-256: f1bbe8bb21286e37...`), and contact sheets (`SHA-256: b658d59290146c5e...`), with `0.00000000` maximum per-cell probability difference.

---

## 5. Known Technical & Scientific Limitations

1. **Severe Source Class Imbalance**: The NIH-NLM full-smear dataset has a `29.6 : 1` Uninfected-to-Parasitized RBC imbalance (`33,071` vs. `1,142`) and only `51` total WBC annotations (`37` in training), preventing reliable single-stage YOLO infection or WBC classification (`WBC mAP@50 = 0%`).
2. **Stage 1 Detection Misses in Crowded Fields**: Across the 25 held-out test images (`4,794` GT RBCs), YOLOv8n localizes `96.28%` (`362 / 376`) of Parasitized RBCs and misses `14` infected cells (`FN_Miss`) in overlapping or low-contrast regions, bounding end-to-end sensitivity.
3. **Residual False Positives on Stain Artifacts**: While Phase 6 domain adaptation reduced raw-crop false positives by `83.4%` (`392` $\rightarrow$ `65`), occasional false positives (`~2.6` per test image) still occur on platelets, Giemsa stain precipitate, or overlapping cell borders.
4. **Single-Center *P. falciparum* Cohort**: Evaluation on `25` test images from `5` held-out patients from a single archive does not establish clinical generalization across diverse hospitals, staining protocols, microscope optics, or other *Plasmodium* species (*P. vivax*, *P. ovale*, *P. malariae*).
5. **Single-Field vs. Multi-Field WHO Protocol**: Clinical microscopy requires examining $\ge 500\text{–}2,000$ RBCs across multiple microscope fields, whereas a single captured field contains $\sim 165\text{–}215$ RBCs.
