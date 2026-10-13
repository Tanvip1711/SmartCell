import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from PIL import Image

BASE_DIR = r"D:\smartcell"
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrix")
MODEL_PATH = os.path.join(MODELS_DIR, "malaria_cnn_best.keras")

for folder in [METRICS_DIR, CM_DIR]:
    os.makedirs(folder, exist_ok=True)

CLASS_NAMES = ["Uninfected", "Parasitized"]
print(f"Loading Model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)

# Collect all test images in deterministic order
records = []
classes_info = [("Uninfected", 0), ("Parasitized", 1)]

for class_name, class_idx in classes_info:
    folder = os.path.join(TEST_DIR, class_name)
    filenames = sorted(os.listdir(folder))
    for fname in filenames:
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            records.append({
                "image_path": os.path.join(folder, fname),
                "actual_class": class_name,
                "actual_label": class_idx
            })

print(f"Total test images verified: {len(records)}")
if len(records) != 4136:
    raise ValueError(f"Expected 4136 test images, but found {len(records)}!")

# Batch prediction
BATCH_SIZE = 64
y_true = [r["actual_label"] for r in records]
y_pred_probs = []

print("Running test predictions...")
for i in range(0, len(records), BATCH_SIZE):
    batch_records = records[i:i + BATCH_SIZE]
    batch_imgs = []
    for r in batch_records:
        img = Image.open(r["image_path"]).convert("RGB").resize((224, 224))
        batch_imgs.append(np.array(img, dtype=np.float32))
    batch_imgs = np.array(batch_imgs)
    preds = model.predict(batch_imgs, verbose=0).flatten()
    y_pred_probs.extend(preds)

y_true = np.array(y_true).astype(int)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs >= 0.5).astype(int)

# 1. Metrics Calculation
test_accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n========================================")
print("SMARTCELL BASELINE CNN TEST EVALUATION")
print("========================================")
print(f"Test Accuracy        : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Precision            : {precision:.4f}")
print(f"Recall (Sensitivity) : {recall:.4f}")
print(f"F1 Score             : {f1:.4f}")
print(f"True Positives  (TP) : {tp}")
print(f"True Negatives  (TN) : {tn}")
print(f"False Positives (FP) : {fp}")
print(f"False Negatives (FN) : {fn}")
print(f"Total Evaluated (TN+TP+FP+FN): {tn + tp + fp + fn}")
print("\nConfusion Matrix:")
print(cm)
print("========================================\n")

if (tn + tp + fp + fn) != 4136:
    raise ValueError(f"Sum of confusion matrix elements ({tn + tp + fp + fn}) != 4136!")

# 2. Save cnn_metrics.json
metrics_dict = {
    "model": "SmartCell_Baseline_CNN",
    "dataset": "NIH Malaria Cell Images",
    "test_set_size": len(y_true),
    "accuracy": float(test_accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "true_positives": int(tp),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn)
}

metrics_json_path = os.path.join(METRICS_DIR, "cnn_metrics.json")
with open(metrics_json_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"Saved metrics JSON to {metrics_json_path}")

# 3. Save cnn_classification_report.txt
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
report_txt_path = os.path.join(METRICS_DIR, "cnn_classification_report.txt")
with open(report_txt_path, "w") as f:
    f.write("SMARTCELL BASELINE CNN CLASSIFICATION REPORT\n")
    f.write("============================================\n\n")
    f.write(report)
print(f"Saved classification report to {report_txt_path}")

# 4. Save Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("SmartCell Baseline CNN - Confusion Matrix", fontsize=12)
plt.colorbar()
tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, fontsize=10)
plt.yticks(tick_marks, CLASS_NAMES, fontsize=10)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight="bold")

plt.ylabel("Actual Label", fontsize=11)
plt.xlabel("Predicted Label", fontsize=11)
plt.tight_layout()
cm_graph_path = os.path.join(CM_DIR, "cnn_confusion_matrix.png")
plt.savefig(cm_graph_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix graph to {cm_graph_path}")

# 5. Save cnn_test_predictions.csv (4,136 rows)
csv_path = os.path.join(METRICS_DIR, "cnn_test_predictions.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_path",
        "actual_class",
        "predicted_class",
        "confidence",
        "uninfected_probability",
        "parasitized_probability"
    ])
    for r, prob, pred in zip(records, y_pred_probs, y_pred):
        parasitized_prob = float(prob)
        uninfected_prob = float(1.0 - parasitized_prob)
        pred_class = "Parasitized" if pred == 1 else "Uninfected"
        conf = parasitized_prob if pred == 1 else uninfected_prob
        writer.writerow([
            r["image_path"],
            r["actual_class"],
            pred_class,
            f"{conf:.4f}",
            f"{uninfected_prob:.4f}",
            f"{parasitized_prob:.4f}"
        ])

print(f"Saved test predictions CSV to {csv_path} ({len(records)} rows)")
