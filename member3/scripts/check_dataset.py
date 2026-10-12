import os
from PIL import Image

base = r"D:\smartcell\cell_images"

classes = ["Parasitized", "Uninfected"]

for class_name in classes:
    folder = os.path.join(base, class_name)

    total = 0
    valid = 0
    corrupted = 0

    print(f"\nChecking: {class_name}")

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        total += 1

        filepath = os.path.join(folder, filename)

        try:
            with Image.open(filepath) as img:
                img.verify()

            valid += 1

        except Exception:
            corrupted += 1
            print("Corrupted:", filename)

    print("Total images :", total)
    print("Valid images :", valid)
    print("Corrupted    :", corrupted)

print("\nDataset check completed.")