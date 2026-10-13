"""
smoke_test_yolo.py
------------------
Executes a 1-epoch smoke test of YOLOv8n on the SmartCell malaria dataset
to verify dataset loading, label caching, model initialization, loss computation,
and validation pipeline before launching the full 50-epoch training run.
"""

import os
import time
import torch
from ultralytics import YOLO

DATA_YAML = r"D:\smartcell\yolo_malaria\data.yaml"
MODEL_WEIGHTS = r"D:\smartcell\yolov8n.pt"
PROJECT_DIR = r"D:\smartcell\results"
RUN_NAME = "yolo_smoke_test"

def main():
    print("=" * 70)
    print("SMARTCELL: YOLOv8n 1-EPOCH SMOKE TEST")
    print("=" * 70)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Selected device: {device}")

    model = YOLO(MODEL_WEIGHTS)

    start_t = time.time()
    results = model.train(
        data=DATA_YAML,
        epochs=1,
        imgsz=640,
        batch=4,
        seed=42,
        workers=0,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        cache="ram",
        verbose=True
    )
    elapsed = time.time() - start_t
    print("=" * 70)
    print(f"SMOKE TEST COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    main()
