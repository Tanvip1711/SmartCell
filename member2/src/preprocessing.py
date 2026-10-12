
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

image_paths = [
    BASE_DIR / "dataset" / "malaria" / "img_001.png",
    BASE_DIR / "dataset" / "all" / "Im001_1.jpg"
]

output_base = BASE_DIR / "results" / "week2"
output_base.mkdir(parents=True, exist_ok=True)

for path in image_paths:
    if not path.exists():
        print("Image not found:", path)
        continue

    image = cv2.imread(str(path))

    if image is None:
        print("Cannot read image:", path)
        continue

    resized = cv2.resize(image, (512, 512))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced = cv2.cvtColor(
        enhanced_lab, cv2.COLOR_LAB2BGR
    )

    filename = path.stem
    dataset_name = path.parent.name.lower()
    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_dir / f"{filename}_resized.png"),
        resized
    )

    cv2.imwrite(
        str(output_dir / f"{filename}_gray.png"),
        gray
    )

    cv2.imwrite(
        str(output_dir / f"{filename}_blurred.png"),
        blurred
    )

    cv2.imwrite(
        str(output_dir / f"{filename}_enhanced.png"),
        enhanced
    )

    original_rgb = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    resized_rgb = cv2.cvtColor(
        resized, cv2.COLOR_BGR2RGB
    )

    enhanced_rgb = cv2.cvtColor(
        enhanced, cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(resized_rgb)
    plt.title("Resized")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(enhanced_rgb)
    plt.title("CLAHE Enhanced")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Processed:", path.name)
    print("Output folder:", output_base)

print("Preprocessing completed.")
