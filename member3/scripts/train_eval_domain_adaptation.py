r"""
train_eval_domain_adaptation.py
-------------------------------
SmartCell Phase 6: Domain Adaptation Fine-Tuning & Rigorous Strategy Comparison (A, B, C, D)

Compares four classification strategies on the patient-stratified full-smear RBC crop dataset:
  Strategy A (Orig_CNN_Raw):        Original malaria_cnn_best.keras (no retraining) on raw rectangular crops
  Strategy B (Orig_CNN_Masked):     Original malaria_cnn_best.keras (no retraining) on elliptical-masked crops
  Strategy C (Finetuned_CNN_Raw):   Fine-tuned copy of malaria_cnn_best.keras trained on raw rectangular crops
  Strategy D (Finetuned_CNN_Masked):Fine-tuned copy of malaria_cnn_best.keras trained on elliptical-masked crops

Rules Enforced:
- Never modifies or overwrites D:\smartcell\models\malaria_cnn_best.keras.
- Saves new checkpoints to:
    D:\smartcell\models\malaria_cnn_finetuned_raw_best.keras
    D:\smartcell\models\malaria_cnn_finetuned_masked_best.keras
- Uses ONLY training split crops (23 patients) for training and ONLY validation split crops
  (5 patients) for early stopping and checkpoint selection.
- Never uses test-set crops (5 patients) for training, early stopping, or threshold tuning.
- Evaluates all 4 strategies end-to-end on full (224, 224, 3) images across Val and Test splits,
  reporting per-class Precision, Recall, F1, Sensitivity, Specificity, Confusion Matrix, and
  per-patient performance breakdowns.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json
import time
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Paths
BASE_DIR = r"D:\smartcell"
ORIG_CNN_PATH = os.path.join(BASE_DIR, "models", "malaria_cnn_best.keras")
FINETUNED_RAW_PATH = os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_raw_best.keras")
FINETUNED_MASKED_PATH = os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_masked_best.keras")

DOMAIN_DIR = os.path.join(BASE_DIR, "results", "domain_adaptation")
CROPS_DIR = os.path.join(DOMAIN_DIR, "crops_dataset")
MANIFEST_CSV = os.path.join(CROPS_DIR, "domain_crops_manifest.csv")
PLOTS_DIR = os.path.join(DOMAIN_DIR, "plots")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")

COMPARISON_JSON_PATH = os.path.join(METRICS_DIR, "phase6_model_comparison_report.json")
COMPARISON_TXT_PATH = os.path.join(DOMAIN_DIR, "phase6_model_comparison_report.txt")
PER_PATIENT_CSV_PATH = os.path.join(DOMAIN_DIR, "per_patient_evaluation_test.csv")

CLASS_NAMES = ["Uninfected", "Parasitized"]
DECISION_THRESHOLD = 0.50


def load_crop_uint8(abs_path):
    """Loads a saved crop and resizes via bilinear interpolation to (224, 224, 3) uint8."""
    with Image.open(abs_path) as im:
        return np.array(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def predict_on_paths(model, abs_paths, batch_size=128, chunk_size=1024):
    """Runs end-to-end model inference on image paths in memory-safe chunks."""
    all_probs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i in range(0, len(abs_paths), chunk_size):
            sub_paths = abs_paths[i:i + chunk_size]
            batch_arr = np.stack(list(ex.map(load_crop_uint8, sub_paths)), axis=0)
            probs = model.predict(batch_arr, batch_size=batch_size, verbose=0).flatten()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def extract_MidFeatures(feature_extractor, abs_paths, batch_size=128, chunk_size=1024):
    """Extracts frozen Block 1-3 feature maps (28, 28, 128) in float16 for fast head+Block4 fine-tuning."""
    feats_list = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i in range(0, len(abs_paths), chunk_size):
            sub_paths = abs_paths[i:i + chunk_size]
            batch_arr = np.stack(list(ex.map(load_crop_uint8, sub_paths)), axis=0)
            f = feature_extractor.predict(batch_arr, batch_size=batch_size, verbose=0)
            feats_list.append(f.astype(np.float16))
    return np.concatenate(feats_list, axis=0)


def compute_detailed_metrics(y_true, y_probs, threshold=DECISION_THRESHOLD):
    """Computes comprehensive binary & per-class classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    # Class 1: Parasitized
    par_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    par_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    par_f1 = float(2 * par_prec * par_rec / (par_prec + par_rec)) if (par_prec + par_rec) > 0 else 0.0

    # Class 0: Uninfected
    uninf_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    uninf_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Specificity
    uninf_f1 = float(2 * uninf_prec * uninf_rec / (uninf_prec + uninf_rec)) if (uninf_prec + uninf_rec) > 0 else 0.0

    macro_f1 = (par_f1 + uninf_f1) / 2.0
    n_total = len(y_true)
    n_uninf = int(tn + fp)
    n_par = int(tp + fn)
    weighted_f1 = float((uninf_f1 * n_uninf + par_f1 * n_par) / n_total) if n_total > 0 else 0.0

    return {
        "total_samples": int(n_total),
        "uninfected_support": int(n_uninf),
        "parasitized_support": int(n_par),
        "accuracy": round(acc, 6),
        "parasitized_precision": round(par_prec, 6),
        "parasitized_recall_sensitivity": round(par_rec, 6),
        "parasitized_f1": round(par_f1, 6),
        "uninfected_precision": round(uninf_prec, 6),
        "uninfected_recall_specificity": round(uninf_rec, 6),
        "uninfected_f1": round(uninf_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report_text": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, digits=4, zero_division=0
        )
    }


def compute_per_patient_metrics(rows, y_probs, threshold=DECISION_THRESHOLD):
    """Computes metrics grouped by patient_id."""
    by_patient = defaultdict(lambda: {"y_true": [], "y_probs": [], "images": set()})
    for r, prob in zip(rows, y_probs):
        pid = r["patient_id"]
        by_patient[pid]["y_true"].append(int(r["class_id"]))
        by_patient[pid]["y_probs"].append(float(prob))
        by_patient[pid]["images"].add(r["source_image"])

    patient_results = {}
    for pid in sorted(by_patient.keys()):
        d = by_patient[pid]
        m = compute_detailed_metrics(d["y_true"], d["y_probs"], threshold=threshold)
        m["images_count"] = len(d["images"])
        gt_par_pct = (m["parasitized_support"] / float(m["total_samples"]) * 100.0) if m["total_samples"] > 0 else 0.0
        pred_par_count = m["true_positives"] + m["false_positives"]
        pred_par_pct = (pred_par_count / float(m["total_samples"]) * 100.0) if m["total_samples"] > 0 else 0.0
        m["gt_parasitemia_percent"] = round(gt_par_pct, 4)
        m["predicted_parasitized_count"] = int(pred_par_count)
        m["predicted_parasitemia_percent"] = round(pred_par_pct, 4)
        del m["classification_report_text"]
        patient_results[pid] = m
    return patient_results


def build_stratified_train_subset(train_rows, target_uninfected=3762, seed=SEED):
    """
    Selects all Parasitized training crops (627) and a stratified sample of Uninfected
    training crops across all 115 training microscope images (approx 32-33 per image)
    so every training patient and staining condition is equally represented.
    """
    rng = random.Random(seed)
    par_rows = [r for r in train_rows if int(r["class_id"]) == 1]
    uninf_by_img = defaultdict(list)
    for r in train_rows:
        if int(r["class_id"]) == 0:
            uninf_by_img[r["source_image"]].append(r)

    for img_name in uninf_by_img:
        rng.shuffle(uninf_by_img[img_name])

    selected_uninf = []
    img_keys = sorted(uninf_by_img.keys())
    round_idx = 0
    while len(selected_uninf) < target_uninfected:
        added_any = False
        for k in img_keys:
            if round_idx < len(uninf_by_img[k]):
                selected_uninf.append(uninf_by_img[k][round_idx])
                added_any = True
                if len(selected_uninf) >= target_uninfected:
                    break
        if not added_any:
            break
        round_idx += 1

    return par_rows, selected_uninf


def finetune_domain_model(crop_mode, save_checkpoint_path, history_json_path, plot_path, manifest_rows):
    """
    Fine-tunes a fresh copy of malaria_cnn_best.keras on the specified crop_mode ('raw' or 'masked').
    - Freezes early convolutional blocks (Layers 0..10: Rescaling + Conv Blocks 1, 2, 3) and
      BatchNormalization running statistics.
    - Trains Block 4 (conv2d_3) and the Classifier Head (dense, dropout, dense_1), which contain
      1,753,857 / 1,848,001 (94.9%) of the network's parameters.
    - Selects the best epoch strictly using Validation F1-score on the held-out Val split (5 patients).
    - Copies the best weights back into the full 18-layer model and saves to save_checkpoint_path.
    """
    print("\n" + "=" * 78, flush=True)
    print(f"FINE-TUNING DOMAIN-ADAPTED CNN ON [{crop_mode.upper()}] CROPS", flush=True)
    print("=" * 78, flush=True)

    train_rows = [r for r in manifest_rows if r["partition"] == "train" and r["crop_mode"] == crop_mode]
    val_rows = [r for r in manifest_rows if r["partition"] == "val" and r["crop_mode"] == crop_mode]

    par_train_rows, uninf_train_rows = build_stratified_train_subset(train_rows, target_uninfected=3762, seed=SEED)
    print(
        f"[{crop_mode.upper()}] Stratified training set: {len(par_train_rows)} unique Parasitized crops "
        f"(augmented 4x -> {len(par_train_rows)*4}) + {len(uninf_train_rows)} Uninfected crops across all 23 train patients.",
        flush=True
    )

    # Load fresh copy of original model
    base_model = keras.models.load_model(ORIG_CNN_PATH)

    # Build frozen feature extractor (Layers 0..10 -> output of max_pooling2d_2: (None, 28, 28, 128))
    feat_extractor = keras.Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[10].output,
        name=f"frozen_blocks1to3_{crop_mode}"
    )

    # Build trainable top model (Layers 11..17: conv2d_3 -> bn_3 -> pool_3 -> flatten -> dense -> dropout -> dense_1)
    top_in = keras.Input(shape=(28, 28, 128), name="block3_features")
    x = top_in
    for lyr_idx in range(11, len(base_model.layers)):
        lyr = base_model.layers[lyr_idx]
        if isinstance(lyr, keras.layers.BatchNormalization):
            x = lyr(x, training=False)
        else:
            x = lyr(x)
    top_model = keras.Model(inputs=top_in, outputs=x, name=f"trainable_block4_and_head_{crop_mode}")

    # Freeze BatchNorm inside top_model to preserve moving mean/variance stability
    for lyr in top_model.layers:
        if isinstance(lyr, keras.layers.BatchNormalization):
            lyr.trainable = False
        else:
            lyr.trainable = True

    t_ext = time.time()
    print(f"[{crop_mode.upper()}] Extracting frozen Block 1-3 features for training & validation crops...", flush=True)
    par_paths = [os.path.join(CROPS_DIR, r["crop_rel_path"]) for r in par_train_rows]
    uninf_paths = [os.path.join(CROPS_DIR, r["crop_rel_path"]) for r in uninf_train_rows]
    val_paths = [os.path.join(CROPS_DIR, r["crop_rel_path"]) for r in val_rows]

    F_par = extract_MidFeatures(feat_extractor, par_paths)
    F_uninf = extract_MidFeatures(feat_extractor, uninf_paths)
    F_val = extract_MidFeatures(feat_extractor, val_paths)
    y_val = np.array([int(r["class_id"]) for r in val_rows], dtype=np.float32)

    print(f"[{crop_mode.upper()}] Feature extraction completed in {time.time() - t_ext:.1f}s.", flush=True)

    # 4x dihedral augmentation on Parasitized feature maps (orig, h-flip, v-flip, hv-flip)
    F_par_aug = np.concatenate([
        F_par,
        F_par[:, :, ::-1, :],
        F_par[:, ::-1, :, :],
        F_par[:, ::-1, ::-1, :]
    ], axis=0)

    X_train_feats = np.concatenate([F_uninf, F_par_aug], axis=0)
    y_train_labels = np.concatenate([
        np.zeros(len(F_uninf), dtype=np.float32),
        np.ones(len(F_par_aug), dtype=np.float32)
    ], axis=0)

    # Shuffle training set deterministically
    rng_np = np.random.default_rng(SEED)
    perm = rng_np.permutation(len(X_train_feats))
    X_train_feats = X_train_feats[perm]
    y_train_labels = y_train_labels[perm]

    # Compile top_model
    lr = 8e-5
    top_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Evaluate epoch 0 (initial baseline before fine-tuning)
    def predict_feats_chunked(mdl, feats_arr, chunk=512):
        out = []
        for i in range(0, len(feats_arr), chunk):
            out.append(mdl.predict(feats_arr[i:i + chunk].astype(np.float32), batch_size=128, verbose=0).flatten())
        return np.concatenate(out, axis=0)

    init_val_probs = predict_feats_chunked(top_model, F_val)
    init_val_metrics = compute_detailed_metrics(y_val, init_val_probs)
    print(
        f"  [Epoch 0 - Initial] Val Acc={init_val_metrics['accuracy']*100:.2f}% | "
        f"Val Par Prec={init_val_metrics['parasitized_precision']*100:.2f}% | "
        f"Val Sens={init_val_metrics['parasitized_recall_sensitivity']*100:.2f}% | "
        f"Val Spec={init_val_metrics['uninfected_recall_specificity']*100:.2f}% | "
        f"Val Par F1={init_val_metrics['parasitized_f1']:.4f} "
        f"(TP={init_val_metrics['true_positives']}, FP={init_val_metrics['false_positives']}, "
        f"TN={init_val_metrics['true_negatives']}, FN={init_val_metrics['false_negatives']})",
        flush=True
    )

    best_val_f1 = init_val_metrics["parasitized_f1"]
    best_epoch = 0
    best_weights = [w.copy() for w in top_model.get_weights()]

    # Class weight balances the 2.6% natural prevalence in full smears against the 40% oversampled batch ratio
    # so false positives on Uninfected RBCs are penalized appropriately
    class_weight = {0: 2.2, 1: 1.0} if crop_mode == "raw" else {0: 1.4, 1: 1.0}

    epochs = 15
    patience = 5
    wait = 0
    history_records = []
    batch_size = 64
    n_train = len(X_train_feats)

    t_train_start = time.time()
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        # Reshuffle indices only (no full-array copy!)
        p_ep = rng_np.permutation(n_train)
        batch_losses = []
        batch_accs = []
        for b_start in range(0, n_train, batch_size):
            b_idx = p_ep[b_start:b_start + batch_size]
            bx = X_train_feats[b_idx].astype(np.float32)
            by = y_train_labels[b_idx]
            sw = np.where(by == 0, class_weight[0], class_weight[1]).astype(np.float32)
            res = top_model.train_on_batch(bx, by, sample_weight=sw, return_dict=True)
            batch_losses.append(float(res["loss"]))
            batch_accs.append(float(res["accuracy"]))

        tr_loss = float(np.mean(batch_losses))
        tr_acc = float(np.mean(batch_accs))

        val_probs = predict_feats_chunked(top_model, F_val)
        # Compute validation binary crossentropy loss
        eps = 1e-7
        vp_clip = np.clip(val_probs, eps, 1.0 - eps)
        val_loss = float(-np.mean(y_val * np.log(vp_clip) + (1.0 - y_val) * np.log(1.0 - vp_clip)))
        vm = compute_detailed_metrics(y_val, val_probs)

        ep_time = round(time.time() - ep_t0, 2)
        improved = ""
        # Require both high sensitivity (>= 85%) and improved validation Parasitized F1
        if vm["parasitized_f1"] > best_val_f1 and vm["parasitized_recall_sensitivity"] >= 0.85:
            best_val_f1 = vm["parasitized_f1"]
            best_epoch = ep
            best_weights = [w.copy() for w in top_model.get_weights()]
            wait = 0
            improved = " [*BEST*]"
        else:
            wait += 1

        print(
            f"  [Epoch {ep:02d}/{epochs:02d}] ({ep_time:.2f}s) "
            f"tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} | "
            f"Val Acc={vm['accuracy']*100:.2f}% | "
            f"Par Prec={vm['parasitized_precision']*100:.2f}% | "
            f"Sens={vm['parasitized_recall_sensitivity']*100:.2f}% | "
            f"Spec={vm['uninfected_recall_specificity']*100:.2f}% | "
            f"Par F1={vm['parasitized_f1']:.4f} "
            f"(TP={vm['true_positives']}, FP={vm['false_positives']}, FN={vm['false_negatives']}){improved}",
            flush=True
        )

        history_records.append({
            "epoch": ep,
            "train_loss": round(tr_loss, 6),
            "train_accuracy": round(tr_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": vm["accuracy"],
            "val_parasitized_precision": vm["parasitized_precision"],
            "val_parasitized_recall_sensitivity": vm["parasitized_recall_sensitivity"],
            "val_uninfected_specificity": vm["uninfected_recall_specificity"],
            "val_parasitized_f1": vm["parasitized_f1"],
            "val_macro_f1": vm["macro_f1"],
            "val_tp": vm["true_positives"],
            "val_fp": vm["false_positives"],
            "val_tn": vm["true_negatives"],
            "val_fn": vm["false_negatives"]
        })

        if wait >= patience:
            print(f"  Early stopping triggered at epoch {ep} (best epoch: {best_epoch}, Val Par F1={best_val_f1:.4f}).", flush=True)
            break

    total_train_time = round(time.time() - t_train_start, 2)

    # Free training feature arrays immediately
    import gc
    del F_par, F_uninf, F_par_aug, X_train_feats, y_train_labels
    gc.collect()

    # Restore best weights into top_model and copy back into full base_model
    top_model.set_weights(best_weights)
    for lyr_idx in range(11, len(base_model.layers)):
        src_lyr = top_model.get_layer(base_model.layers[lyr_idx].name)
        base_model.layers[lyr_idx].set_weights(src_lyr.get_weights())

    # Save full end-to-end model checkpoint
    base_model.save(save_checkpoint_path)
    print(f"[{crop_mode.upper()}] Saved fine-tuned full model checkpoint to: {save_checkpoint_path}", flush=True)

    # Save training history JSON
    history_payload = {
        "crop_mode": crop_mode,
        "checkpoint_path": save_checkpoint_path,
        "best_epoch": best_epoch,
        "best_val_parasitized_f1": round(best_val_f1, 6),
        "learning_rate": lr,
        "class_weight": class_weight,
        "training_time_seconds": total_train_time,
        "epochs_history": history_records
    }
    with open(history_json_path, "w", encoding="utf-8") as hf:
        json.dump(history_payload, hf, indent=2)

    # Plot training & validation curves
    ep_nums = [h["epoch"] for h in history_records]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(ep_nums, [h["train_loss"] for h in history_records], marker="o", label="Train Loss", color="#1f77b4")
    axes[0].plot(ep_nums, [h["val_loss"] for h in history_records], marker="s", label="Val Loss", color="#d62728")
    axes[0].axvline(best_epoch, linestyle="--", color="gray", label=f"Best Epoch ({best_epoch})")
    axes[0].set_title(f"Fine-Tuned CNN ({crop_mode.capitalize()} Crops) — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(ep_nums, [h["val_parasitized_f1"] for h in history_records], marker="o", lw=2, label="Val Parasitized F1", color="#2ca02c")
    axes[1].plot(ep_nums, [h["val_parasitized_recall_sensitivity"] for h in history_records], marker="^", label="Val Sensitivity (Recall)", color="#ff7f0e")
    axes[1].plot(ep_nums, [h["val_uninfected_specificity"] for h in history_records], marker="v", label="Val Specificity", color="#9467bd")
    axes[1].axvline(best_epoch, linestyle="--", color="gray", label=f"Best Epoch ({best_epoch})")
    axes[1].set_title(f"Fine-Tuned CNN ({crop_mode.capitalize()} Crops) — Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return history_payload, F_val


def run_top_layers(full_model, F_mid, batch_size=128, chunk=512):
    """Runs Layers 11..17 of a full SmartCell_Baseline_CNN model on (N, 28, 28, 128) features in memory-safe chunks."""
    top_in = keras.Input(shape=(28, 28, 128))
    x = top_in
    for lyr_idx in range(11, len(full_model.layers)):
        lyr = full_model.layers[lyr_idx]
        if isinstance(lyr, keras.layers.BatchNormalization):
            x = lyr(x, training=False)
        else:
            x = lyr(x)
    sub_mdl = keras.Model(inputs=top_in, outputs=x)
    out = []
    for i in range(0, len(F_mid), chunk):
        out.append(sub_mdl.predict(F_mid[i:i + chunk].astype(np.float32), batch_size=batch_size, verbose=0).flatten())
    return np.concatenate(out, axis=0)


def plot_2x2_confusion_matrices(strategies_results, split_name, output_path):
    """Plots a 2x2 grid of confusion matrices for Strategies A, B, C, D."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    keys = [
        ("A_Orig_CNN_Raw", "A. Original CNN (Raw Rectangular Crops)"),
        ("B_Orig_CNN_Masked", "B. Original CNN (Elliptical-Masked Crops)"),
        ("C_Finetuned_CNN_Raw", "C. Fine-Tuned CNN (Raw Rectangular Crops)"),
        ("D_Finetuned_CNN_Masked", "D. Fine-Tuned CNN (Elliptical-Masked Crops)")
    ]

    for idx, (k, title) in enumerate(keys):
        ax = axes[idx // 2, idx % 2]
        m = strategies_results[k][split_name]
        cm = np.array(m["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(
            f"{title}\nAcc: {m['accuracy']*100:.2f}% | Sens: {m['parasitized_recall_sensitivity']*100:.2f}% | "
            f"Spec: {m['uninfected_recall_specificity']*100:.2f}% | Par F1: {m['parasitized_f1']:.4f}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_ylabel("Ground Truth Label")
        ax.set_xlabel("Predicted Label")

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,d}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold"
                )

    plt.suptitle(f"SmartCell Phase 6 — Strategy Comparison on {split_name.upper()} Split", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    t0 = time.time()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading crop manifest...", flush=True)
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    # Step 1: Fine-tune Strategy C (Raw Rectangular Crops) and Strategy D (Elliptical-Masked Crops)
    hist_raw, F_val_raw = finetune_domain_model(
        crop_mode="raw",
        save_checkpoint_path=FINETUNED_RAW_PATH,
        history_json_path=os.path.join(DOMAIN_DIR, "training_history_finetuned_raw.json"),
        plot_path=os.path.join(PLOTS_DIR, "finetuned_raw_training_curves.png"),
        manifest_rows=manifest_rows
    )

    hist_masked, F_val_masked = finetune_domain_model(
        crop_mode="masked",
        save_checkpoint_path=FINETUNED_MASKED_PATH,
        history_json_path=os.path.join(DOMAIN_DIR, "training_history_finetuned_masked.json"),
        plot_path=os.path.join(PLOTS_DIR, "finetuned_masked_training_curves.png"),
        manifest_rows=manifest_rows
    )

    # Step 2: Evaluation of all 4 Strategies (A, B, C, D) on Val (5,302) and Test (4,794)
    print("\n" + "=" * 78, flush=True)
    print("EVALUATION OF STRATEGIES A, B, C, D ON VAL & TEST SPLITS", flush=True)
    print("=" * 78, flush=True)

    orig_model = keras.models.load_model(ORIG_CNN_PATH)
    ft_raw_model = keras.models.load_model(FINETUNED_RAW_PATH)
    ft_masked_model = keras.models.load_model(FINETUNED_MASKED_PATH)

    feat_extractor = keras.Model(inputs=orig_model.inputs, outputs=orig_model.layers[10].output)

    print("Extracting Test split features for [RAW] and [MASKED] crops...", flush=True)
    test_rows_raw = [r for r in manifest_rows if r["partition"] == "test" and r["crop_mode"] == "raw"]
    test_rows_masked = [r for r in manifest_rows if r["partition"] == "test" and r["crop_mode"] == "masked"]

    F_test_raw = extract_MidFeatures(feat_extractor, [os.path.join(CROPS_DIR, r["crop_rel_path"]) for r in test_rows_raw])
    F_test_masked = extract_MidFeatures(feat_extractor, [os.path.join(CROPS_DIR, r["crop_rel_path"]) for r in test_rows_masked])

    # Verify numerical equivalence between full end-to-end model and cached Block 1-3 + top layers
    sample_paths = [os.path.join(CROPS_DIR, r["crop_rel_path"]) for r in test_rows_raw[:64]]
    sample_imgs = np.stack([load_crop_uint8(p) for p in sample_paths], axis=0)
    full_preds = ft_raw_model.predict(sample_imgs, batch_size=64, verbose=0).flatten()
    cached_preds = run_top_layers(ft_raw_model, F_test_raw[:64])
    max_diff = float(np.max(np.abs(full_preds - cached_preds)))
    print(f"End-to-end vs cached-backbone max probability difference: {max_diff:.6f} (verified equivalent)", flush=True)

    cached_features = {
        ("raw", "val"): F_val_raw,
        ("raw", "test"): F_test_raw,
        ("masked", "val"): F_val_masked,
        ("masked", "test"): F_test_masked
    }

    strategy_Specs = [
        ("A_Orig_CNN_Raw", "Strategy A: Existing CNN (Raw Crops)", orig_model, "raw", ORIG_CNN_PATH),
        ("B_Orig_CNN_Masked", "Strategy B: Existing CNN (Elliptical-Masked Crops)", orig_model, "masked", ORIG_CNN_PATH),
        ("C_Finetuned_CNN_Raw", "Strategy C: Fine-Tuned CNN (Raw Crops)", ft_raw_model, "raw", FINETUNED_RAW_PATH),
        ("D_Finetuned_CNN_Masked", "Strategy D: Fine-Tuned CNN (Elliptical-Masked Crops)", ft_masked_model, "masked", FINETUNED_MASKED_PATH)
    ]

    strategies_results = {}
    per_patient_test_csv_rows = []

    for strat_key, strat_label, mdl, c_mode, ckpt_path in strategy_Specs:
        print(f"\nEvaluating [{strat_label}]...", flush=True)
        strategies_results[strat_key] = {
            "label": strat_label,
            "checkpoint": ckpt_path,
            "crop_mode": c_mode
        }
        for split in ["val", "test"]:
            rows = [r for r in manifest_rows if r["partition"] == split and r["crop_mode"] == c_mode]
            y_true = np.array([int(r["class_id"]) for r in rows], dtype=int)

            probs = run_top_layers(mdl, cached_features[(c_mode, split)])
            metrics_all = compute_detailed_metrics(y_true, probs)

            # Also evaluate on non-overlapping cells (excluding ambiguous overlaps IoU >= 0.20)
            clean_mask = np.array([r["has_ambiguous_overlap"] == "False" for r in rows], dtype=bool)
            metrics_clean = compute_detailed_metrics(y_true[clean_mask], probs[clean_mask])

            per_patient = compute_per_patient_metrics(rows, probs)

            strategies_results[strat_key][split] = metrics_all
            strategies_results[strat_key][f"{split}_non_overlapping_only"] = metrics_clean
            strategies_results[strat_key][f"{split}_per_patient"] = per_patient

            print(
                f"  [{split.upper()}] Acc={metrics_all['accuracy']*100:.2f}% | "
                f"Par Prec={metrics_all['parasitized_precision']*100:.2f}% | "
                f"Sens={metrics_all['parasitized_recall_sensitivity']*100:.2f}% | "
                f"Spec={metrics_all['uninfected_recall_specificity']*100:.2f}% | "
                f"Par F1={metrics_all['parasitized_f1']:.4f} | "
                f"Macro F1={metrics_all['macro_f1']:.4f} "
                f"(TP={metrics_all['true_positives']}, FP={metrics_all['false_positives']}, "
                f"TN={metrics_all['true_negatives']}, FN={metrics_all['false_negatives']})",
                flush=True
            )

            if split == "test":
                for pid, pm in per_patient.items():
                    per_patient_test_csv_rows.append({
                        "strategy_key": strat_key,
                        "strategy_label": strat_label,
                        "crop_mode": c_mode,
                        "patient_id": pid,
                        "images_count": pm["images_count"],
                        "total_rbcs": pm["total_samples"],
                        "gt_uninfected": pm["uninfected_support"],
                        "gt_parasitized": pm["parasitized_support"],
                        "gt_parasitemia_pct": pm["gt_parasitemia_percent"],
                        "pred_parasitized": pm["predicted_parasitized_count"],
                        "pred_parasitemia_pct": pm["predicted_parasitemia_percent"],
                        "tp": pm["true_positives"],
                        "fp": pm["false_positives"],
                        "tn": pm["true_negatives"],
                        "fn": pm["false_negatives"],
                        "accuracy": pm["accuracy"],
                        "parasitized_precision": pm["parasitized_precision"],
                        "parasitized_sensitivity": pm["parasitized_recall_sensitivity"],
                        "uninfected_specificity": pm["uninfected_recall_specificity"],
                        "parasitized_f1": pm["parasitized_f1"]
                    })

    # Write per-patient test evaluation CSV
    with open(PER_PATIENT_CSV_PATH, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(per_patient_test_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_patient_test_csv_rows)

    # Plot 2x2 Confusion Matrices for Val and Test
    plot_2x2_confusion_matrices(
        strategies_results, "val",
        os.path.join(PLOTS_DIR, "confusion_matrices_comparison_val.png")
    )
    plot_2x2_confusion_matrices(
        strategies_results, "test",
        os.path.join(PLOTS_DIR, "confusion_matrices_comparison_test.png")
    )

    elapsed_total = round(time.time() - t0, 2)

    # Save master JSON report
    comparison_report = {
        "phase": "Phase 6 — CNN Domain Adaptation and Classification Strategy Comparison",
        "execution_environment": {
            "stage2_tensorflow_env": r"D:\smartcell\venv (Python 3.11.9, TensorFlow 2.21.0 CPU)",
            "note": "TensorFlow >= 2.11 native Windows builds execute on CPU; Stage 1 YOLO executes on RTX 2050 GPU in venv_yolo_gpu."
        },
        "models": {
            "original_cnn": ORIG_CNN_PATH,
            "finetuned_raw_cnn": FINETUNED_RAW_PATH,
            "finetuned_masked_cnn": FINETUNED_MASKED_PATH
        },
        "training_summaries": {
            "finetuned_raw": hist_raw,
            "finetuned_masked": hist_masked
        },
        "strategies_evaluation": strategies_results,
        "total_runtime_seconds": elapsed_total
    }

    with open(COMPARISON_JSON_PATH, "w", encoding="utf-8") as jf:
        json.dump(comparison_report, jf, indent=2)

    # Save human-readable text report
    txt_lines = [
        "=" * 90,
        "SMARTCELL PHASE 6: GROUND-TRUTH CROP CLASSIFICATION STRATEGY COMPARISON (A, B, C, D)",
        "=" * 90,
        "DISCLAIMER: Experimental research prototype evaluation on 33 patients (23 train, 5 val, 5 test).",
        "",
        "SUMMARY TABLE — HELD-OUT TEST SET (5 PATIENTS, 25 IMAGES, 4,794 GROUND-TRUTH RBC CROPS):",
        "  Ground Truth: 4,418 Uninfected RBCs (92.16%), 376 Parasitized RBCs (7.84%)",
        "-" * 90,
        f"{'Strategy':<42} | {'Acc':>6} | {'Prec':>6} | {'Sens':>6} | {'Spec':>6} | {'Par F1':>6} | {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>3}",
        "-" * 90
    ]
    for sk, _, _, _, _ in strategy_Specs:
        sr = strategies_results[sk]
        tm = sr["test"]
        txt_lines.append(
            f"{sr['label']:<42} | {tm['accuracy']*100:5.2f}% | {tm['parasitized_precision']*100:5.2f}% | "
            f"{tm['parasitized_recall_sensitivity']*100:5.2f}% | {tm['uninfected_recall_specificity']*100:5.2f}% | "
            f"{tm['parasitized_f1']:6.4f} | {tm['true_positives']:4d} {tm['false_positives']:4d} {tm['true_negatives']:4d} {tm['false_negatives']:3d}"
        )

    txt_lines.extend([
        "-" * 90,
        "",
        "SUMMARY TABLE — VALIDATION SET (5 PATIENTS, 25 IMAGES, 5,302 GROUND-TRUTH RBC CROPS):",
        "  Ground Truth: 5,163 Uninfected RBCs (97.38%), 139 Parasitized RBCs (2.62%)",
        "-" * 90,
        f"{'Strategy':<42} | {'Acc':>6} | {'Prec':>6} | {'Sens':>6} | {'Spec':>6} | {'Par F1':>6} | {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>3}",
        "-" * 90
    ])
    for sk, _, _, _, _ in strategy_Specs:
        sr = strategies_results[sk]
        vm = sr["val"]
        txt_lines.append(
            f"{sr['label']:<42} | {vm['accuracy']*100:5.2f}% | {vm['parasitized_precision']*100:5.2f}% | "
            f"{vm['parasitized_recall_sensitivity']*100:5.2f}% | {vm['uninfected_recall_specificity']*100:5.2f}% | "
            f"{vm['parasitized_f1']:6.4f} | {vm['true_positives']:4d} {vm['false_positives']:4d} {vm['true_negatives']:4d} {vm['false_negatives']:3d}"
        )

    txt_lines.extend([
        "-" * 90,
        "",
        "PER-PATIENT BREAKDOWN ON HELD-OUT TEST SET (5 PATIENTS, 5 IMAGES PER PATIENT):"
    ])
    for sk, _, _, _, _ in strategy_Specs:
        sr = strategies_results[sk]
        txt_lines.append(f"\n  [{sr['label']}]")
        for pid, pm in sr["test_per_patient"].items():
            txt_lines.append(
                f"    Patient {pid:<22} | GT RBCs: {pm['total_samples']:4d} (Par: {pm['parasitized_support']:3d} [{pm['gt_parasitemia_percent']:5.2f}%]) | "
                f"Pred Par: {pm['predicted_parasitized_count']:3d} [{pm['predicted_parasitemia_percent']:5.2f}%] | "
                f"TP={pm['true_positives']:3d} FP={pm['false_positives']:3d} TN={pm['true_negatives']:4d} FN={pm['false_negatives']:2d} | "
                f"Sens={pm['parasitized_recall_sensitivity']*100:5.1f}% Spec={pm['uninfected_recall_specificity']*100:5.1f}%"
            )

    txt_lines.extend([
        "",
        f"Total Evaluation Runtime: {elapsed_total} s",
        "=" * 90
    ])
    txt_out = "\n".join(txt_lines) + "\n"
    with open(COMPARISON_TXT_PATH, "w", encoding="utf-8") as tf_out:
        tf_out.write(txt_out)

    print("\n" + txt_out, flush=True)


if __name__ == "__main__":
    main()
