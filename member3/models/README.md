# SmartCell — Member 3: Model Checkpoints & Setup Guide

To keep the GitHub repository lightweight and avoid committing large binary weights (`~98.8 MB` total across `.keras` and `.pt` checkpoints), the trained model checkpoints are stored locally in `D:\smartcell` and excluded from Git version control.

## Preserved Model Checkpoints Inventory

| Model Role | Local Path in `D:\smartcell` | Filename | Size (Bytes) | Framework / Format |
| :--- | :--- | :--- | :---: | :--- |
| **Stage 1 YOLOv8n Detector (Best)** | `D:\smartcell\results\yolo_malaria_run\weights\best.pt` | `best.pt` | `6,227,690` (~5.94 MB) | Ultralytics PyTorch (`YOLOv8n`) |
| **Stage 1 YOLOv8n Detector (Last)** | `D:\smartcell\results\yolo_malaria_run\weights\last.pt` | `last.pt` | `6,227,690` (~5.94 MB) | Ultralytics PyTorch (`YOLOv8n`) |
| **Base YOLOv8n Pretrained Weights** | `D:\smartcell\yolov8n.pt` | `yolov8n.pt` | `6,549,796` (~6.25 MB) | Ultralytics PyTorch (`YOLOv8n`) |
| **Stage 2 Baseline CNN (Phase 1)** | `D:\smartcell\models\malaria_cnn_best.keras` | `malaria_cnn_best.keras` | `22,257,316` (~21.23 MB) | TensorFlow / Keras 3 (`.keras`) |
| **Stage 2 MobileNetV2 (Phase 2)** | `D:\smartcell\models\malaria_mobilenetv2_best.keras` | `malaria_mobilenetv2_best.keras` | `19,307,715` (~18.41 MB) | TensorFlow / Keras 3 (`.keras`) |
| **Stage 2 Domain-Adapted Raw CNN (Phase 6 — Recommended)** | `D:\smartcell\models\malaria_cnn_finetuned_raw_best.keras` | `malaria_cnn_finetuned_raw_best.keras` | `22,256,757` (~21.23 MB) | TensorFlow / Keras 3 (`.keras`) |
| **Stage 2 Domain-Adapted Masked CNN (Phase 6)** | `D:\smartcell\models\malaria_cnn_finetuned_masked_best.keras` | `malaria_cnn_finetuned_masked_best.keras` | `22,256,757` (~21.23 MB) | TensorFlow / Keras 3 (`.keras`) |
| **Class Index Mapping (Included in Git)** | `D:\smartcell\models\malaria_class_names.json` | `malaria_class_names.json` | `46` | JSON (`{"0": "Uninfected", "1": "Parasitized"}`) |

## How to Reproduce or Place Checkpoints Locally

1. **Place Existing Checkpoints**:
   - Copy the `.keras` files into `D:\smartcell\models\`.
   - Copy `best.pt` into `D:\smartcell\results\yolo_malaria_run\weights\best.pt`.
2. **Or Regenerate via Training Scripts**:
   - **Phase 1 Baseline CNN**: `D:\smartcell\venv\Scripts\python.exe scripts/train_cnn.py`
   - **Phase 2 MobileNetV2**: `D:\smartcell\venv\Scripts\python.exe scripts/train_mobilenetv2.py`
   - **Phase 4 YOLOv8n Detector**: `D:\smartcell\venv_yolo\Scripts\python.exe scripts/train_yolo_malaria.py`
   - **Phase 6 Domain-Adapted CNNs (`finetuned_raw` & `finetuned_masked`)**:
     ```powershell
     D:\smartcell\venv\Scripts\python.exe scripts/prepare_domain_crops.py
     D:\smartcell\venv\Scripts\python.exe scripts/train_eval_domain_adaptation.py
     ```
