@echo off
title SmartCell Phase 7 - Two-Stage YOLOv8n (GPU) + CNN Malaria Screening Prototype
echo ==============================================================================
echo SMARTCELL INTERACTIVE DEMONSTRATION APPLICATION (RESEARCH PROTOTYPE ONLY)
echo ==============================================================================
echo Stage 1 Detector:   D:\smartcell\venv_yolo_gpu (NVIDIA GeForce RTX 2050 CUDA:0)
echo Stage 2 Classifier: D:\smartcell\venv (TensorFlow 2.21.0 CPU)
echo Web Interface URL:  http://127.0.0.1:8501
echo ==============================================================================
start "" http://127.0.0.1:8501
"D:\smartcell\venv\Scripts\python.exe" "D:\smartcell\scripts\smartcell_demo_app.py" --host 127.0.0.1 --port 8501
pause
