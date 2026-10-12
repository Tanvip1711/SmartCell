
import cv2
import csv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

dataset_dirs = [
    BASE_DIR / "dataset" / "malaria",
    BASE_DIR / "dataset" / "all"
]

output_base = (
    BASE_DIR / "results" / "week2" / "evaluation"
)

output_base.mkdir(parents=True, exist_ok=True)

MIN_AREA = 100
MAX_AREA = 20000
MIN_SIDE = 5
MAX_ASPECT_RATIO = 5.0
MIN_CIRCULARITY = 0.05

extensions = {".png", ".jpg", ".jpeg", ".bmp"}


def get_candidates(mask):
    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_SIDE or h < MIN_SIDE:
            continue

        aspect_ratio = max(w, h) / max(min(w, h), 1)

        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = (
                4 * 3.14159265 * area / (perimeter ** 2)
            )
        else:
            circularity = 0

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        if circularity < MIN_CIRCULARITY:
            continue

        valid.append({
            "contour": contour,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity
        })

    return valid


for dataset_dir in dataset_dirs:

    if not dataset_dir.exists():
        print("Missing dataset:", dataset_dir)
        continue

    dataset_name = dataset_dir.name.lower()

    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "evaluation_report.csv"

    image_paths = [
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not image_paths:
        print("No images found:", dataset_dir)
        continue

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "image_name",
            "polarity",
            "candidate_id",
            "area",
            "x",
            "y",
            "width",
            "height",
            "aspect_ratio",
            "circularity"
        ])

        for path in image_paths:

            image = cv2.imread(str(path))

            if image is None:
                print("Cannot read:", path.name)
                continue

            image = cv2.resize(image, (512, 512))

            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            )

            blurred = cv2.GaussianBlur(
                gray, (5, 5), 0
            )

            _, otsu = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            normal_candidates = get_candidates(otsu)
            inverse_candidates = get_candidates(
                cv2.bitwise_not(otsu)
            )

            if len(inverse_candidates) > len(normal_candidates):
                mask = cv2.bitwise_not(otsu)
                candidates = inverse_candidates
                polarity = "inverted"
            else:
                mask = otsu
                candidates = normal_candidates
                polarity = "normal"

            annotated = image.copy()

            for index, region in enumerate(candidates, start=1):

                x = region["x"]
                y = region["y"]
                w = region["w"]
                h = region["h"]

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    str(index),
                    (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

                writer.writerow([
                    path.name,
                    polarity,
                    index,
                    round(region["area"], 2),
                    x,
                    y,
                    w,
                    h,
                    round(region["aspect_ratio"], 3),
                    round(region["circularity"], 3)
                ])

            annotated_path = (
                output_dir / f"{path.stem}_filtered.png"
            )

            cv2.imwrite(str(annotated_path), annotated)

            print(
                f"{dataset_name} | {path.name} | "
                f"Candidates: {len(candidates)} | "
                f"Polarity: {polarity}"
            )

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Selected Binary Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            )
            plt.title("Filtered Candidates")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    print("Report saved:", csv_path)
    print("Annotated images saved:", output_dir)

print("Day 5 evaluation completed.")
