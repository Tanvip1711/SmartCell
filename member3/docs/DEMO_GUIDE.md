# SmartCell — College Project Live Demonstration Guide (5–7 Minute Sequence)

> **RESEARCH PROTOTYPE DISCLAIMER**  
> **SmartCell** is an experimental academic research prototype developed for a college engineering project. It is **not** a clinically validated medical diagnostic device and must never be used as a substitute for professional microscopic examination or clinical diagnosis.

---

## 1. Executive Presentation Summary: Two-Stage Architecture & Dual-Environment Design

When introducing the system to evaluators, use this concise architectural explanation (approx. 45 seconds):

1. **Why a Two-Stage Pipeline (`YOLOv8n` $\rightarrow$ `Custom CNN`)?**
   - **Stage 1 — Candidate Cell Localization (`YOLOv8n`)**: Full-resolution thin blood-smear images ($5312 \times 2988\text{ px}$) contain $150\text{–}300$ closely packed cells per microscope field. On the NIH-NLM Thin Blood Smears dataset (`33` patients, `165` images, `34,264` polygons), YOLOv8n achieves **96.44% $\text{mAP@50}$** on Uninfected RBCs and localizes **97.93% (`4,695 / 4,794`)** of all ground-truth RBCs on the 25 held-out test images, but struggles to directly classify rare *Plasmodium falciparum* ring forms (`2.07%` class-specific $\text{mAP@50}$ due to a `29.6 : 1` class imbalance).
   - **Stage 2 — High-Specificity Infection Classification (`Custom CNN`)**: Instead of trusting YOLO's class label for infection status, Stage 1 acts as a class-agnostic erythrocyte localizer (excluding `class 2 [WBC]` detections and suppressing duplicate boxes at $\text{IoU} \ge 0.50$). Each native-resolution RBC crop ($+4\text{ px}$ context padding) is passed to our dedicated Stage 2 CNN classifier (`malaria_cnn_finetuned_raw_best.keras`), achieving **98.35% accuracy**, **96.28% sensitivity**, **98.53% specificity**, and **0.9016 Parasitized F1** on ground-truth full-smear test crops (and **1.12% Mean Absolute Error** on end-to-end parasitemia across all 25 test images).

2. **Why Separate GPU (`venv_yolo_gpu`) and CPU (`venv`) Environments?**
   - **Native Windows Framework Compatibility**: TensorFlow $\ge 2.11$ (`TensorFlow 2.21.0` in `D:\smartcell\venv`) dropped native Windows CUDA GPU support (requiring WSL2 for GPU execution), whereas PyTorch `2.14.1+cu130` (`D:\smartcell\venv_yolo_gpu`) natively supports the laptop's **NVIDIA GeForce RTX 2050 GPU (`CUDA device=0`, 4 GB VRAM)**.
   - **Zero-Conflict Isolation**: Keeping `venv_yolo_gpu` (PyTorch + Ultralytics on CUDA:0) and `venv` (TensorFlow 2.21.0 + Keras on CPU) strictly isolated avoids `numpy`/`protobuf`/`typing-extensions` dependency conflicts and guarantees deterministic reproduction of every phase checkpoint. The orchestrator bridges both environments seamlessly via structured JSON and PNG crop artifacts in ~16 seconds per full-resolution blood smear.

---

## 2. Pre-Demo Checklist (30 Seconds Before Evaluators Arrive)

1. Open a terminal or File Explorer at `D:\smartcell`.
2. Confirm no other service is occupying port `8501`.
3. Keep these two sample test images ready to mention (or upload directly from `D:\smartcell\yolo_malaria\images\test\`):
   - **High Parasitemia Smear**: `D:\smartcell\yolo_malaria\images\test\IMG_20150819_133447.jpg` (Held-out test Patient `209C68P29N_ThinF`; Ground Truth: `71` Parasitized / `135` Uninfected / `3` WBCs = `206` RBCs, **`34.47%` parasitemia**).
   - **Healthy Uninfected Control Smear**: `D:\smartcell\yolo_malaria\images\test\IMG_20151207_114038.jpg` (Held-out test Patient `377C238NThinF`; Ground Truth: `0` Parasitized / `208` Uninfected = **`0.00%` parasitemia**).

---

## 3. Step-by-Step 5–7 Minute Live Demonstration Sequence

### Minute 0:00 – 0:45 | Step 1: Launch the Application & Verify Hardware Status
1. Double-click **`D:\smartcell\launch_smartcell_demo.bat`** (or run in PowerShell):
   ```powershell
   D:\smartcell\venv\Scripts\python.exe D:\smartcell\scripts\smartcell_demo_app.py --host 127.0.0.1 --port 8501
   ```
2. Open **`http://127.0.0.1:8501`** in the browser.
3. **Point out to evaluators**:
   - The red **Research Prototype Notice** banner at the top confirming this is an academic screening prototype, not a clinical device.
   - The live hardware badges in the top-right header:
     - `Stage 1 GPU: NVIDIA GeForce RTX 2050 (3299.7/4095.5 MB VRAM)`
     - `Stage 2: TensorFlow 2.21.0 (venv CPU)`

---

### Minute 0:45 – 1:45 | Step 2: Upload a Full-Resolution Blood-Smear Image & Explain Stage 1 YOLO Detection
1. In **Card 1 (Blood-Smear Image Input)** on the left sidebar, click **Choose File** and upload:
   `D:\smartcell\yolo_malaria\images\test\IMG_20150819_133447.jpg`
   *(Alternatively, select `IMG_20150819_133447.jpg` from the **Preset Held-Out Test Smear** dropdown).*
2. **Explain Stage 1 while configuring**:
   - The input image is a raw $5312 \times 2988\text{ px}$ thin blood smear microscope field.
   - Stage 1 runs YOLOv8n (`best.pt`) on the RTX 2050 GPU (`device=0`) with confidence threshold `0.25`, NMS IoU `0.45`, and class-agnostic deduplication IoU `0.50`.
   - Any detected `class 2 (WBC)` boxes are separated and excluded from the RBC count, and any tiny/clipped artifacts ($< 16\text{ px}$) are rejected before Stage 2.

---

### Minute 1:45 – 2:45 | Step 3: Select CNN Variant, Crop Mode & Thresholds (Domain Adaptation Demo)
1. In **Card 2 (Stage 2 Classifier & Crop Mode)**:
   - Show the three selectable CNN checkpoints:
     1. **`Fine-Tuned Raw CNN (Phase 6 Domain-Adapted — Recommended)`** (`malaria_cnn_finetuned_raw_best.keras`)
     2. **`Fine-Tuned Masked CNN (Phase 6 Elliptical-Masked)`** (`malaria_cnn_finetuned_masked_best.keras`)
     3. **`Original Baseline CNN (Phase 1 Single-Cell Dataset)`** (`malaria_cnn_best.keras`)
   - Keep **`Fine-Tuned Raw CNN`** and **`raw — Rectangular RGB Crop (4px padding)`** selected.
2. **Briefly explain the Phase 6 Domain Adaptation breakthrough**:
   - The original Phase 1 CNN was trained on segmented single cells with pure black `[0, 0, 0]` backgrounds, so bright pink plasma corners and neighboring cell margins in raw rectangular full-smear crops caused false positives (`108 / 195 = 55.38%` on this image).
   - Fine-tuning on `24,112` domain-matched rectangular crops from the 23 training patients resolved this corner sensitivity, cutting false positives by **83.4%** across the test set without needing artificial corner masking.

---

### Minute 2:45 – 4:15 | Step 4: Run Analysis, Explain Separate Counts, Parasitemia & Confidence Scores
1. Click **"Run Two-Stage Blood-Smear Analysis"** (`~16 seconds` total: `~6.8s` GPU Stage 1 + `~9.3s` CPU Stage 2).
2. Walk through the **6 Summary KPI Cards** at the top of the right panel:
   - **Predicted Parasitized RBCs**: **`74`**
   - **Predicted Uninfected RBCs**: **`121`** (out of **`195`** valid candidate RBC crops)
   - **Experimental Parasitemia**: **`37.95%`** (`74 / 195 classified RBCs`), closely tracking the ground-truth `34.47%` (`71 / 206` RBCs).
   - **Excluded WBC Detections**: **`0`** (shown separately so leukocytes never distort parasitemia).
   - **Rejected / Duplicate Boxes**: **`0`** (shown separately).
   - **Processing Time**: **`~16.1s`** (`S1: ~6.8s` on RTX 2050 using only **`23.73 MB` Peak VRAM** | `S2: ~9.3s` on CPU).
3. Point out the **Active Configuration & Confidence Telemetry** panel:
   - **YOLO Detection Confidence**: Mean **`63.6%`** (`25.0%–92.3%`).
   - **CNN Overall Confidence**: Mean **`97.2%`**, Median **`100.0%`**.
   - **Audit Comparison**: Shows both the **Raw Crop (`37.95%`, `74` Par)** and **Elliptical-Masked Crop (`36.41%`, `71` Par)** estimates side by side.

---

### Minute 4:15 – 5:15 | Step 5: Inspect Visual Outputs & Download CSV / JSON Exports
1. **Tab 1 — Annotated Full-Smear Image**:
   - Show the full microscope field with **Red bounding boxes + confidence labels** on predicted Parasitized RBCs and **Green bounding boxes** on Uninfected RBCs.
2. **Tab 2 — Candidate RBC Contact Sheet**:
   - Show the 25-cell high-resolution zoomed grid (`contact_sheet_IMG_20150819_133447.jpg`) where intracellular ring trophozoites inside `cell_0001`, `cell_0003`, `cell_0008`, `cell_0009`, `cell_0011` are clearly visible.
3. **Tab 3 — Per-Cell Predictions & Confidence (`195` cells)**:
   - Filter by **"Parasitized RBCs Only"** to inspect individual cell thumbnails, `[x1, y1, x2, y2]` coordinates, YOLO confidence, and CNN probabilities.
4. **Click the Export Buttons**:
   - Click **Download Per-Cell CSV (`cell_predictions_IMG_20150819_133447.csv`)** and **Download Summary JSON (`field_summary_IMG_20150819_133447.json`)** to demonstrate that every cell prediction and field summary is exported cleanly for downstream research analysis.

---

### Minute 5:15 – 6:15 | Step 6: Demonstrate Edge-Case Handling (Zero Detections or Invalid Box / WBC Exclusion)
1. **Option A — Zero-Detection Edge Case (`2 seconds`)**:
   - Check **"Simulate Zero YOLO Detections (`--test-zero-detections`)"** in Card 3 and click **Run Two-Stage Blood-Smear Analysis**.
   - Point out that the application safely handles `0` detections without division-by-zero, displaying `0 / 0 (Zero-denominator handled safely: no valid candidate RBCs detected)` and `0.00%` parasitemia.
2. **Option B — WBC Exclusion & Invalid Box Rejection**:
   - Uncheck Zero Detections, check **"Inject Synthetic WBC & Invalid Boxes (`--inject-test-invalid-boxes`)"**, and run analysis.
   - Open **Tab 4 (Excluded WBCs & Rejected Boxes)** to show `1` excluded `WBC (ID 2)` detection (`[500, 500, 720, 720]`) and `2` rejected invalid/out-of-bounds boxes (`5x4 px < 16 px` and `0x0 px < 16 px`), while the `195` valid RBCs still yield the exact same `74` Parasitized / `121` Uninfected counts.
3. **Option C — Invalid / Corrupted File Upload**:
   - Upload any non-image or corrupted file to show the structured red `InvalidImageError` banner (`HTTP 400`) without server interruption.

---

### Minute 6:15 – 7:00 | Step 7: Conclude with Honest Scientific Limitations & Future Work
Wrap up the presentation by highlighting five engineering and scientific takeaways:
1. **Severe Source Class Imbalance**: The NIH-NLM full-smear dataset has a `29.6 : 1` Uninfected-to-Parasitized ratio (`33,071` vs. `1,142` RBCs) and only `51` WBCs total (`37` in training), which is why a two-stage architecture (`YOLO` localization + domain-adapted `CNN` classification) dramatically outperforms single-stage YOLO classification (`0.8698` end-to-end Parasitized F1 vs. `2.07%` YOLO Parasitized $\text{mAP@50}$).
2. **Stage 1 Detection Misses in Crowded Fields**: Across all 25 test images (`4,794` GT RBCs), YOLOv8n localizes `96.28%` (`362 / 376`) of Parasitized RBCs and misses `14` overlapping/faint cells (`FN_Miss`), bounding end-to-end recall at `96.28%`.
3. **Limited Held-Out Patient Cohort**: The test split comprises `25` images from `5` held-out patients from a single *P. falciparum* thin-smear archive; multi-center validation across diverse Giemsa staining protocols and *Plasmodium* species (*P. vivax*, *P. malariae*, *P. ovale*) is required for broader generalization.
4. **Single-Field vs. Clinical Multi-Field Counting**: WHO microscopy protocols require counting $\ge 500\text{–}2,000$ RBCs across multiple fields, whereas a single microscope image contains $\sim 165\text{–}215$ RBCs.
5. **Hardware & Memory Constraints**: On the 4 GB RTX 2050 laptop GPU, full-resolution ($5312 \times 2988$) inference is run at `batch=1` (`~24–30 MB` peak VRAM) for stability.
