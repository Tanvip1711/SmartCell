
import cv2
import csv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

dataset_dirs = [
    BASE_DIR / "dataset" / "malaria",
    BASE_DIR / "dataset" / "all"
]

segmentation_base = (
    BASE_DIR / "results" / "week2" / "segmentation"
)

output_base = (
    BASE_DIR / "results" / "week2" / "contours"
)

extensions = {".png", ".jpg", ".jpeg", ".bmp"}

MIN_AREA = 100
MAX_AREA = 50000

for dataset_dir in dataset_dirs:

    if not dataset_dir.exists():
        print("Dataset folder not found:", dataset_dir)
        continue

    dataset_name = dataset_dir.name.lower()

    output_dir = output_base / dataset_name
    crops_dir = output_dir / "crops"

    crops_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not image_paths:
        print("No images found in:", dataset_dir)
        continue

    csv_path = output_dir / "regions.csv"

    with open(csv_path, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "image_name",
            "region_id",
            "x",
            "y",
            "width",
            "height",
            "area",
            "crop_path"
        ])

        for path in image_paths:

            image = cv2.imread(str(path))

            if image is None:
                print("Cannot read:", path.name)
                continue

            resized = cv2.resize(image, (512, 512))

            mask_path = (
                segmentation_base
                / dataset_name
                / f"{path.stem}_otsu.png"
            )

            if mask_path.exists():
                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if mask is None:
                    print("Cannot read mask:", mask_path)
                    continue

                mask = cv2.resize(mask, (512, 512))

            else:
                gray = cv2.cvtColor(
                    resized, cv2.COLOR_BGR2GRAY
                )

                blurred = cv2.GaussianBlur(
                    gray, (5, 5), 0
                )

                _, mask = cv2.threshold(
                    blurred,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            contours, _ = cv2.findContours(
                mask.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            annotated = resized.copy()
            region_count = 0

            for contour in contours:

                area = cv2.contourArea(contour)

                if area < MIN_AREA or area > MAX_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                if w < 5 or h < 5:
                    continue

                region_count += 1

                crop = resized[y:y + h, x:x + w]

                crop_name = (
                    f"{path.stem}_region_{region_count}.png"
                )

                crop_path = crops_dir / crop_name

                cv2.imwrite(str(crop_path), crop)

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    str(region_count),
                    (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

                writer.writerow([
                    path.name,
                    region_count,
                    x,
                    y,
                    w,
                    h,
                    round(area, 2),
                    str(crop_path.relative_to(BASE_DIR))
                ])

            annotated_path = (
                output_dir / f"{path.stem}_detected.png"
            )

            cv2.imwrite(str(annotated_path), annotated)

            original_rgb = cv2.cvtColor(
                resized, cv2.COLOR_BGR2RGB
            )

            annotated_rgb = cv2.cvtColor(
                annotated, cv2.COLOR_BGR2RGB
            )

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(original_rgb)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Otsu Segmentation Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(annotated_rgb)
            plt.title("Candidate Regions")
            plt.axis("off")

            plt.suptitle(
                f"{dataset_name.upper()}: {path.name}"
            )

            plt.tight_layout()
            plt.show()

            print(
                f"{path.name}: "
                f"{region_count} candidate regions"
            )

    print("CSV saved:", csv_path)
    print("Images and crops saved in:", output_dir)

print("Day 4 contour detection completed.")
