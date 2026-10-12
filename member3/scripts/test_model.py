import os
import sys
import glob
import random
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def test_random_samples(num_samples=10, model_path=None):
    base_dir = r"D:\smartcell"
    test_dir = os.path.join(base_dir, "dataset", "test")
    
    if model_path is None:
        model_path = os.path.join(base_dir, "models", "malaria_cnn_best.keras")
        if not os.path.exists(model_path):
            mobilenet_path = os.path.join(base_dir, "models", "malaria_mobilenetv2_best.keras")
            if os.path.exists(mobilenet_path):
                model_path = mobilenet_path
            else:
                print(f"Error: Model file not found at '{model_path}'.")
                sys.exit(1)

    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow not installed.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Collect images strictly from test set
    classes = ["Uninfected", "Parasitized"]
    samples = []
    for cls in classes:
        cls_folder = os.path.join(test_dir, cls)
        img_paths = glob.glob(os.path.join(cls_folder, "*.png"))
        selected = random.sample(img_paths, min(num_samples // 2, len(img_paths)))
        for p in selected:
            samples.append((p, cls))

    random.shuffle(samples)

    print("\n=========================================================================================")
    print("SMARTCELL MODEL TEST SAMPLE PREDICTIONS (TEST SET ONLY)")
    print("=========================================================================================")
    print(f"{'Image Name':<35} | {'Actual Class':<15} | {'Predicted Class':<15} | {'Confidence':<10} | {'Status'}")
    print("-" * 89)

    correct = 0
    for path, actual_class in samples:
        img = Image.open(path).convert("RGB").resize((224, 224))
        img_arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        prob = float(model.predict(img_arr, verbose=0)[0][0])

        if prob >= 0.5:
            pred_class = "Parasitized"
            confidence = prob
        else:
            pred_class = "Uninfected"
            confidence = 1.0 - prob

        is_correct = (pred_class == actual_class)
        if is_correct:
            correct += 1
        status_str = "MATCH" if is_correct else "MISMATCH"

        img_basename = os.path.basename(path)
        print(f"{img_basename:<35} | {actual_class:<15} | {pred_class:<15} | {confidence * 100:>8.2f}% | {status_str}")

    print("=========================================================================================")
    print(f"Sample Accuracy: {correct} / {len(samples)} ({correct / len(samples) * 100:.2f}%)")
    print("=========================================================================================\n")

if __name__ == "__main__":
    test_random_samples(num_samples=10)
