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

tf.keras.utils.set_random_seed(42)

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

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
HEAD_EPOCHS = 3
FINE_TUNE_EPOCHS = 2
CLASS_NAMES = ["Uninfected", "Parasitized"]

print("--- Loading Datasets for MobileNetV2 ---")
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

# Realistic Fast Augmentation for Training Only
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomBrightness(0.05),
], name="mobilenet_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
train_ds_proc = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
val_ds_proc = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds_proc = test_ds.prefetch(buffer_size=AUTOTUNE)

# Build Transfer Learning Model
def build_mobilenetv2_model():
    inputs = keras.Input(shape=(224, 224, 3))
    # MobileNetV2 expects [-1, 1] input range
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
    
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze base model initially
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs, name="SmartCell_MobileNetV2")
    return model, base_model

model, base_model = build_mobilenetv2_model()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model_save_path = os.path.join(MODELS_DIR, "malaria_mobilenetv2_best.keras")
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

print("\n--- Phase 1: Training Classification Head (Backbone Frozen) ---")
start_time = time.time()
history_head = model.fit(
    train_ds_proc,
    validation_data=val_ds_proc,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks
)

print("\n--- Phase 2: Fine-Tuning Selected Top Layers of MobileNetV2 ---")
# Unfreeze top 20 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with smaller learning rate for fine tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_ds_proc,
    validation_data=val_ds_proc,
    epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=len(history_head.history["accuracy"]),
    callbacks=callbacks
)
total_training_time = round(time.time() - start_time, 2)
print(f"MobileNetV2 training completed in {total_training_time} seconds.")

# Combine histories
combined_acc = history_head.history["accuracy"] + history_finetune.history["accuracy"]
combined_val_acc = history_head.history["val_accuracy"] + history_finetune.history["val_accuracy"]
combined_loss = history_head.history["loss"] + history_finetune.history["loss"]
combined_val_loss = history_head.history["val_loss"] + history_finetune.history["val_loss"]

history_data = {
    "accuracy": combined_acc,
    "val_accuracy": combined_val_acc,
    "loss": combined_loss,
    "val_loss": combined_val_loss
}
history_path = os.path.join(METRICS_DIR, "mobilenet_history.json")
with open(history_path, "w") as f:
    json.dump(history_data, f, indent=4)

# Plot accuracy & loss curves
epochs_range = range(1, len(combined_acc) + 1)

# Accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, combined_acc, label="Training Accuracy", color="#1f77b4", lw=2)
plt.plot(epochs_range, combined_val_acc, label="Validation Accuracy", color="#ff7f0e", lw=2)
plt.title("SmartCell MobileNetV2 - Training & Validation Accuracy", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy", fontsize=11)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
acc_graph_path = os.path.join(GRAPHS_DIR, "mobilenet_accuracy.png")
plt.savefig(acc_graph_path, dpi=300, bbox_inches="tight")
plt.close()

# Loss graph
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, combined_loss, label="Training Loss", color="#1f77b4", lw=2)
plt.plot(epochs_range, combined_val_loss, label="Validation Loss", color="#d62728", lw=2)
plt.title("SmartCell MobileNetV2 - Training & Validation Loss", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Binary Crossentropy Loss", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)
loss_graph_path = os.path.join(GRAPHS_DIR, "mobilenet_loss.png")
plt.savefig(loss_graph_path, dpi=300, bbox_inches="tight")
plt.close()

# Evaluate on strictly isolated Test Set
print("\n--- Evaluating MobileNetV2 on Untouched Test Set ---")
best_model = keras.models.load_model(model_save_path)

y_pred_probs = best_model.predict(test_ds_proc, verbose=1).flatten()
y_true = np.concatenate([labels.numpy().flatten() for _, labels in test_ds_proc]).astype(int)
y_pred = (y_pred_probs >= 0.5).astype(int)

test_accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n==========================================")
print("SMARTCELL MOBILENETV2 TEST EVALUATION")
print("==========================================")
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
print("==========================================")

# Save Metrics JSON
metrics_dict = {
    "model": "SmartCell_MobileNetV2",
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
    "training_time_seconds": float(total_training_time),
    "epochs_trained": len(combined_acc)
}

metrics_json_path = os.path.join(METRICS_DIR, "mobilenet_metrics.json")
with open(metrics_json_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)

# Save Classification Report
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
report_txt_path = os.path.join(METRICS_DIR, "mobilenet_classification_report.txt")
with open(report_txt_path, "w") as f:
    f.write("SMARTCELL MOBILENETV2 CLASSIFICATION REPORT\n")
    f.write("===========================================\n\n")
    f.write(report)

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
plt.title("SmartCell MobileNetV2 - Confusion Matrix", fontsize=12)
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
cm_graph_path = os.path.join(CM_DIR, "mobilenet_confusion_matrix.png")
plt.savefig(cm_graph_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix graph to {cm_graph_path}")
