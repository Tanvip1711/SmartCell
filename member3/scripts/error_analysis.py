import os
import sys
import csv
import shutil
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def run_error_analysis(model_path=None, max_save_examples=25):
    base_dir = r"D:\smartcell"
    test_dir = os.path.join(base_dir, "dataset", "test")
    
    if model_path is None:
        model_path = os.path.join(base_dir, "models", "malaria_cnn_best.keras")
        if not os.path.exists(model_path):
            mobilenet_path = os.path.join(base_dir, "models", "malaria_mobilenetv2_best.keras")
            if os.path.exists(mobilenet_path):
                model_path = mobilenet_path
            else:
                print(f"Error: Model not found at '{model_path}'.")
                sys.exit(1)

    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow not installed.")
        sys.exit(1)

    fn_dir = os.path.join(base_dir, "results", "predictions", "false_negatives")
    fp_dir = os.path.join(base_dir, "results", "predictions", "false_positives")
    os.makedirs(fn_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, "results", "metrics", "error_analysis.csv")

    csv_preds_path = os.path.join(base_dir, "results", "metrics", "cnn_test_predictions.csv")
    error_records = []
    fn_saved = 0
    fp_saved = 0
    total_evaluated = 0

    if os.path.exists(csv_preds_path):
        print(f"Loading predictions directly from {csv_preds_path}...")
        with open(csv_preds_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_evaluated += 1
                img_path = row["image_path"]
                actual = row["actual_class"]
                pred = row["predicted_class"]
                conf = row["confidence"]
                fname = os.path.basename(img_path)

                if actual != pred:
                    err_type = "False Negative" if (actual == "Parasitized" and pred == "Uninfected") else "False Positive"
                    error_records.append({
                        "image_path": img_path,
                        "actual_class": actual,
                        "predicted_class": pred,
                        "confidence": conf,
                        "error_type": err_type
                    })
                    if err_type == "False Negative" and fn_saved < max_save_examples:
                        dst = os.path.join(fn_dir, f"FN_{fname}")
                        shutil.copy2(img_path, dst)
                        fn_saved += 1
                    elif err_type == "False Positive" and fp_saved < max_save_examples:
                        dst = os.path.join(fp_dir, f"FP_{fname}")
                        shutil.copy2(img_path, dst)
                        fp_saved += 1
    else:
        print(f"Running error analysis with model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        classes = [("Uninfected", 0), ("Parasitized", 1)]
        records = []
        for class_name, class_idx in classes:
            folder = os.path.join(test_dir, class_name)
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    records.append({
                        "image_path": os.path.join(folder, fname),
                        "actual_class": class_name,
                        "actual_label": class_idx,
                        "fname": fname
                    })
        
        batch_size = 64
        y_pred_probs = []
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            imgs = [np.array(Image.open(r["image_path"]).convert("RGB").resize((224, 224)), dtype=np.float32) for r in batch]
            probs = model.predict(np.array(imgs), verbose=0).flatten()
            y_pred_probs.extend(probs)
        
        total_evaluated = len(records)
        for r, prob in zip(records, y_pred_probs):
            pred_label = 1 if prob >= 0.5 else 0
            pred_class = "Parasitized" if pred_label == 1 else "Uninfected"
            conf = float(prob if pred_label == 1 else (1.0 - prob))
            true_label = r["actual_label"]
            actual_class = r["actual_class"]
            fname = r["fname"]
            img_path = r["image_path"]

            if pred_label != true_label:
                err_type = "False Negative" if (true_label == 1 and pred_label == 0) else "False Positive"
                error_records.append({
                    "image_path": img_path,
                    "actual_class": actual_class,
                    "predicted_class": pred_class,
                    "confidence": f"{conf:.4f}",
                    "error_type": err_type
                })
                if err_type == "False Negative" and fn_saved < max_save_examples:
                    dst = os.path.join(fn_dir, f"FN_{fname}")
                    shutil.copy2(img_path, dst)
                    fn_saved += 1
                elif err_type == "False Positive" and fp_saved < max_save_examples:
                    dst = os.path.join(fp_dir, f"FP_{fname}")
                    shutil.copy2(img_path, dst)
                    fp_saved += 1

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "actual_class", "predicted_class", "confidence", "error_type"])
        writer.writeheader()
        writer.writerows(error_records)

    print("\n==========================================")
    print("SMARTCELL ERROR ANALYSIS COMPLETED")
    print("==========================================")
    print(f"Total Test Images Evaluated: {total_evaluated}")
    print(f"Total Errors Found         : {len(error_records)}")
    print(f"False Negatives Saved      : {fn_saved} (in {fn_dir})")
    print(f"False Positives Saved      : {fp_saved} (in {fp_dir})")
    print(f"Detailed Error Log         : {csv_path}")
    print("==========================================\n")

if __name__ == "__main__":
    run_error_analysis()
