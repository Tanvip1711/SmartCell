"""
train_yolo_malaria.py
---------------------
Phase 4 Baseline YOLOv8n Object Detection Training for SmartCell.

Configuration:
- Model: yolov8n.pt (pretrained)
- Dataset: D:\smartcell\yolo_malaria\data.yaml
- Epochs: 50
- Image Size: 640 (rectangular 16:9 -> 640x384 preserving native 5312x2988 aspect ratio)
- Batch Size: 4
- Max Detections: 350 (dataset images contain up to 313 cells/image)
- Random Seed: 42
- Workers: 0 (Windows multiprocessing safe)
- Cache: RAM
- Output: D:\smartcell\results\yolo_malaria_run
"""

import os
import time
import json
import torch
import ultralytics
from ultralytics import YOLO

DATA_YAML = r"D:\smartcell\yolo_malaria\data.yaml"
MODEL_WEIGHTS = r"D:\smartcell\yolov8n.pt"
PROJECT_DIR = r"D:\smartcell\results"
RUN_NAME = "yolo_malaria_run"

def main():
    print("=" * 70)
    print("SMARTCELL PHASE 4: YOLOv8n BASELINE TRAINING (50 EPOCHS)")
    print("=" * 70)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Python Environment: {os.sys.executable}")
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"Ultralytics:        {ultralytics.__version__}")
    print(f"Execution Device:   {device}")
    print("=" * 70)

    model = YOLO(MODEL_WEIGHTS)

    start_time = time.time()
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=4,
        seed=42,
        workers=0,
        device=device,
        rect=True,
        max_det=350,
        cache="ram",
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )
    elapsed_sec = time.time() - start_time

    timing_info = {
        "training_duration_seconds": round(elapsed_sec, 2),
        "training_duration_minutes": round(elapsed_sec / 60.0, 2),
        "training_duration_hours": round(elapsed_sec / 3600.0, 4),
        "device": device,
        "epochs": 50,
        "imgsz": 640,
        "batch": 4,
        "rect": True,
        "max_det": 350,
        "seed": 42
    }

    timing_path = os.path.join(PROJECT_DIR, RUN_NAME, "training_timing.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=2)

    print("=" * 70)
    print(f"TRAINING COMPLETED SUCCESSFULLY in {elapsed_sec:.2f}s ({elapsed_sec/60.0:.2f} min)")
    print(f"Saved timing metadata to: {timing_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
