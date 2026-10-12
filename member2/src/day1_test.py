
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

image_paths = [
    Path("dataset/malaria/img_001.png"),
    Path("dataset/all/Im001_1.jpg")
]

for path in image_paths:
    if not path.exists():
        print(f"Image not found: {path}")
        continue

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
