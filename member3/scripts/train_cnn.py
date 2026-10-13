import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(42)

# Paths
BASE_DIR = r"D:\smartcell"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrix")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

for folder in [MODELS_DIR, GRAPHS_DIR, CM_DIR, METRICS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 2e-4

# Class names mapping: 0 = Uninfected, 1 = Parasitized
CLASS_NAMES = ["Uninfected", "Parasitized"]
print(f"Class mapping: 0 -> {CLASS_NAMES[0]}, 1 -> {CLASS_NAMES[1]}")

print("\n--- Loading Datasets ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Realistic Augmentation for Training Only
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomBrightness(0.05)
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def prepare_train_ds(ds):
    return ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

def prepare_val_test_ds(ds):
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds_proc = prepare_train_ds(train_ds)
val_ds_proc = prepare_val_test_ds(val_ds)
test_ds_proc = prepare_val_test_ds(test_ds)

# Baseline CNN Architecture (~1.8M parameters)
def build_baseline_cnn():
    inputs = keras.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Dense Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="SmartCell_Baseline_CNN")
    return model

model = build_baseline_cnn()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
model_save_path = os.path.join(MODELS_DIR, "malaria_cnn_best.keras")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("\n--- Training Custom Baseline CNN ---")
start_time = time.time()
history = model.fit(
    train_ds_proc,
    validation_data=val_ds_proc,
    epochs=EPOCHS,
    callbacks=callbacks
)
training_time = round(time.time() - start_time, 2)
print(f"Training completed in {training_time} seconds.")

# Save training history
history_save_path = os.path.join(METRICS_DIR, "cnn_history.json")
with open(history_save_path, "w") as f:
    json.dump(history.history, f, indent=4)
print(f"Saved training history to {history_save_path}")

# Plot Accuracy & Loss Graphs
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

# Accuracy Graph
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, label="Training Accuracy", color="#1f77b4", lw=2)
plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#ff7f0e", lw=2)
plt.title("SmartCell Baseline CNN - Training and Validation Accuracy", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy", fontsize=11)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
acc_graph_path = os.path.join(GRAPHS_DIR, "cnn_accuracy.png")
plt.savefig(acc_graph_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved accuracy graph to {acc_graph_path}")

# Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, loss, label="Training Loss", color="#1f77b4", lw=2)
plt.plot(epochs_range, val_loss, label="Validation Loss", color="#d62728", lw=2)
plt.title("SmartCell Baseline CNN - Training and Validation Loss", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Binary Crossentropy Loss", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)
loss_graph_path = os.path.join(GRAPHS_DIR, "cnn_loss.png")
plt.savefig(loss_graph_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved loss graph to {loss_graph_path}")

# Evaluation on strictly isolated Test Set
print("\n--- Evaluating Best Model on Untouched Test Set ---")
best_model = keras.models.load_model(model_save_path)

y_pred_probs = best_model.predict(test_ds_proc, verbose=1).flatten()
y_true = np.concatenate([labels.numpy().flatten() for _, labels in test_ds_proc]).astype(int)
y_pred = (y_pred_probs >= 0.5).astype(int)

# Metrics calculation
test_accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n========================================")
print("SMARTCELL BASELINE CNN TEST EVALUATION")
print("========================================")
print(f"Test Accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"True Positives  (TP) : {tp}")
print(f"True Negatives  (TN) : {tn}")
print(f"False Positives (FP) : {fp}")
print(f"False Negatives (FN) : {fn}")
print("\nConfusion Matrix:")
print(cm)
print("========================================")

# Save Metrics JSON
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
    "false_negatives": int(fn),
    "training_time_seconds": float(training_time),
    "epochs_trained": len(acc)
}

metrics_json_path = os.path.join(METRICS_DIR, "cnn_metrics.json")
with open(metrics_json_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"Saved metrics JSON to {metrics_json_path}")

# Save Classification Report
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print("\nClassification Report:\n", report)

report_txt_path = os.path.join(METRICS_DIR, "cnn_classification_report.txt")
with open(report_txt_path, "w") as f:
    f.write("SMARTCELL BASELINE CNN CLASSIFICATION REPORT\n")
    f.write("============================================\n\n")
    f.write(report)
print(f"Saved classification report to {report_txt_path}")

# Plot Confusion Matrix Heatmap
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
