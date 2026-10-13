import os
import sys

# Environment variables must be set before importing TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
from PIL import Image
import tensorflow as tf

def predict_single_image(image_path, model_path=None):
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at '{image_path}'")
        sys.exit(1)

    base_dir = r"D:\smartcell"
    if model_path is None:
        model_path = os.path.join(base_dir, "models", "malaria_cnn_best.keras")
        if not os.path.exists(model_path):
            mobilenet_path = os.path.join(base_dir, "models", "malaria_mobilenetv2_best.keras")
            if os.path.exists(mobilenet_path):
                model_path = mobilenet_path
            else:
                print(f"Error: Model file not found at '{model_path}'. Please train the model first.")
                sys.exit(1)

    # Class mapping: 0 = Uninfected, 1 = Parasitized (matches training pipeline)
    class_names_path = os.path.join(base_dir, "models", "malaria_class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        class_names = {"0": "Uninfected", "1": "Parasitized"}

    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        sys.exit(1)

    # Load and preprocess image exactly as during training:
    # 224x224 RGB, model contains internal Rescaling layer (1.0/255)
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        sys.exit(1)

    # Prediction (binary sigmoid output corresponds to index 1: Parasitized)
    prediction = float(model.predict(img_batch, verbose=0)[0][0])
    parasitized_prob = prediction
    uninfected_prob = 1.0 - parasitized_prob

    if parasitized_prob >= 0.5:
        predicted_class = "Parasitized"
        confidence = parasitized_prob
        status_str = "Suspicious for Malaria"
    else:
        predicted_class = "Uninfected"
        confidence = uninfected_prob
        status_str = "No Malaria Parasite Detected"

    # Display single standardized research report
    print("========================================")
    print("SMARTCELL")
    print("SINGLE IMAGE PREDICTION")
    print("========================================")
    print(f"\nImage:\n{os.path.abspath(image_path)}")
    print(f"\nPrediction:\n{predicted_class}")
    print(f"\nConfidence:\n{confidence * 100:.2f}%")
    print("\nProbabilities:")
    print(f"Parasitized: {parasitized_prob * 100:.2f}% ({parasitized_prob:.4f})")
    print(f"Uninfected: {uninfected_prob * 100:.2f}% ({uninfected_prob:.4f})")
    print(f"\nStatus:\n{status_str}")
    print("\n========================================")
    print("RESEARCH PROTOTYPE ONLY")
    print("NOT A CERTIFIED MEDICAL DIAGNOSIS")
    print("========================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_malaria.py <image_path> [model_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    mod_path = sys.argv[2] if len(sys.argv) > 2 else None

    predict_single_image(img_path, mod_path)
