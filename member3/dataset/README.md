# SmartCell — Member 3: Dataset Configuration, Label Metadata & Split Documentation

> **Note on Repository Storage**: In accordance with project repository guidelines, the complete raw image datasets (`cell_images/`, `Polygon Set/`, `Point Set/`, `dataset/{train,val,test}` single-cell images, `yolo_malaria/images/` full-smear `.jpg` images) and generated domain-adaptation crop collections (`results/domain_adaptation/crops_dataset/`) are excluded from Git version control due to size (`> 2.5 GB`). Only the dataset configuration (`data.yaml`), official statistics (`Dataset_statistics.xlsx`), split verification reports (`yolo_preparation_report.json` / `.txt`), and the 165 normalized YOLO label `.txt` metadata files (`yolo_malaria/labels/{train,val,test}`) are included here.

---

## 1. Dataset 1: NIH Malaria Single-Cell Classification Dataset (`cell_images` & `dataset/`)

- **Source**: Official NIH NLM Malaria Screener Segmented Single-Cell Dataset (`27,558` RGB cell crops).
- **Quality Verification**:
  - Corrupted image scan (`scripts/check_dataset.py`): **`0` corrupted files** across `27,558` images.
  - Exact MD5 duplicate scan (`scripts/check_duplicates.py`): **`0` duplicate files**.
- **Balanced Train / Validation / Test Split (`scripts/split_dataset.py`, Seed `42`, `70% / 15% / 15%`)**:

| Partition | Parasitized | Uninfected | Total Images | Share |
| :--- | :---: | :---: | :---: | :---: |
| **Train (`dataset/train`)** | `9,645` | `9,645` | **`19,290`** | `70.0%` |
| **Validation (`dataset/val`)** | `2,066` | `2,066` | **`4,132`** | `15.0%` |
| **Test (`dataset/test`)** | `2,068` | `2,068` | **`4,136`** | `15.0%` |
| **Total** | **`13,779`** | **`13,779`** | **`27,558`** | **`100.0%`** |

---

## 2. Dataset 2: NIH-NLM Thin Blood Smears *Plasmodium falciparum* Full-Field Dataset (`Polygon Set` $\rightarrow$ `yolo_malaria/`)

- **Source**: Official NIH-NLM Thin Blood Smears *Pf* (`Polygon Set`), comprising `165` high-resolution ($5312 \times 2988\text{ px}$) microscopic fields across `33` patients (`5` images per patient) and `34,264` polygonal cell annotations.
- **Conversion & Verification (`scripts/prepare_yolo_malaria.py` & `scripts/verify_yolo_labels.py`)**:
  - Converted polygon vertices into normalized YOLO bounding boxes (`class_id x_center y_center width height`).
  - Verified **`0` invalid or out-of-bounds annotations** and **`0` degenerate bounding boxes**.
- **Class Mapping (`data.yaml`)**:
  - `0`: `Uninfected RBC` — `33,071` instances (`96.52%`)
  - `1`: `Parasitized RBC` — `1,142` instances (`3.33%`)
  - `2`: `WBC` — `51` instances (`0.15%`)
- **Strict Patient-Level Stratified Split (Seed `42`, Zero Patient Leakage)**:

| Partition | Patients | Microscopic Fields | Total Annotated Cells | Uninfected RBC (`0`) | Parasitized RBC (`1`) | WBC (`2`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (`labels/train`)** | `23` (`69.7%`) | `115` | `24,154` (`70.5%`) | `23,490` | `627` | `37` |
| **Validation (`labels/val`)** | `5` (`15.2%`) | `25` | `5,310` (`15.5%`) | `5,163` | `139` | `8` |
| **Test (`labels/test`)** | `5` (`15.2%`) | `25` | `4,800` (`14.0%`) | `4,418` | `376` | `6` |
| **Total** | **`33`** | **`165`** | **`34,264`** | **`33,071`** | **`1,142`** | **`51`** |

- **Held-Out Test Patients (`5` Patients, `25` Images)**:
  - `130C192P153NThinF` (`5` images, `933` Uninfected, `150` Parasitized, `13.85%` GT parasitemia)
  - `209C68P29N_ThinF` (`5` images, `669` Uninfected, `224` Parasitized, `25.08%` GT parasitemia)
  - `358C180P141NThinF` (`5` images, `899` Uninfected, `1` Parasitized, `0.11%` GT parasitemia)
  - `372C187P148NThinF` (`5` images, `974` Uninfected, `1` Parasitized, `0.10%` GT parasitemia)
  - `377C238NThinF` (`5` images, `943` Uninfected, `0` Parasitized, `0.00%` GT parasitemia — uninfected control)

---

## 3. Dataset 3: Phase 6 Domain-Adapted Ground-Truth RBC Crop Dataset (`results/domain_adaptation/crops_dataset/`)

Generated via `scripts/prepare_domain_crops.py` from `yolo_malaria` ground-truth RBC annotations (`classes 0` and `1` only; all `51` `class 2 [WBC]` annotations excluded) using the exact same `23 / 5 / 5` patient split (`0` patient overlap):
- **Train**: `24,112` RBC crops (`23,485` Uninfected, `627` Parasitized)
- **Validation**: `5,302` RBC crops (`5,163` Uninfected, `139` Parasitized)
- **Test**: `4,794` RBC crops (`4,418` Uninfected, `376` Parasitized)
- **Total**: `34,208` Raw rectangular crops (`+4 px` context padding) + `34,208` Elliptical-Masked crops (`[0, 0, 0]` black corners).
- Full summary metadata is included in `../results/metrics/phase6_domain_crops_report.json` and `../results/domain_adaptation/phase6_domain_crops_report.txt`.
