
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

dataset_dirs = [
    BASE_DIR / "dataset" / "malaria",
    BASE_DIR / "dataset" / "all"
]

output_base = BASE_DIR / "results" / "week2" / "segmentation"
output_base.mkdir(parents=True, exist_ok=True)

extensions = {".png", ".jpg", ".jpeg", ".bmp"}

for dataset_dir in dataset_dirs:

    if not dataset_dir.exists():
        print("Dataset folder not found:", dataset_dir)
        continue

    output_dir = output_base / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not image_paths:
        print("No images found in:", dataset_dir)
        continue

    for path in image_paths:

        image = cv2.imread(str(path))

        if image is None:
            print("Cannot read image:", path.name)
            continue

        resized = cv2.resize(image, (512, 512))

        gray = cv2.cvtColor(
            resized, cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray, (5, 5), 0
        )

        # Otsu thresholding
        _, otsu = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5
        )

        # Morphological operations on Otsu mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )

        opening = cv2.morphologyEx(
            otsu, cv2.MORPH_OPEN, kernel
        )

        closing = cv2.morphologyEx(
            opening, cv2.MORPH_CLOSE, kernel
        )

        filename = path.stem

        outputs = {
            "gray": gray,
            "otsu": otsu,
            "adaptive": adaptive,
            "opening": opening,
            "morphology": closing
        }

        for name, result in outputs.items():
            save_path = output_dir / f"{filename}_{name}.png"
            cv2.imwrite(str(save_path), result)

        original_rgb = cv2.cvtColor(
            resized, cv2.COLOR_BGR2RGB
        )

        plt.figure(figsize=(14, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(original_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(otsu, cmap="gray")
        plt.title("Otsu Thresholding")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(adaptive, cmap="gray")
        plt.title("Adaptive Thresholding")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(closing, cmap="gray")
        plt.title("Morphological Processing")
        plt.axis("off")

        plt.suptitle(
            f"{dataset_dir.name.upper()} - {path.name}"
        )

        plt.tight_layout()
        plt.show()

        print("Processed:", path.name)
        print("Saved to:", output_dir)

print("Day 3 segmentation completed.")
