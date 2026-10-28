
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

malaria_path = BASE_DIR / "dataset" / "malaria" / "img_001.png"
all_path = BASE_DIR / "dataset" / "all" / "Im001_1.jpg"

for path in [malaria_path, all_path]:
    image = cv2.imread(str(path))

    if image is None:
        print(f"Image not found or could not be read: {path}")
    else:
        print(f"Image loaded successfully: {path.name}")
        print(f"Image dimensions: {image.shape}")

    image = cv2.imread(str(path))

    if image is None:
        print(f"Could not read: {path}")
        continue

    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    print("File:", path.name)
    print("Resolution:", width, "x", height)
    print("Channels:", channels)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 5))
    plt.imshow(rgb_image)
    plt.title(path.name)
    plt.axis("off")
    plt.show()
