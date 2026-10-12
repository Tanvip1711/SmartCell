import os
import shutil
import random

# Original dataset
SOURCE = r"D:\smartcell\cell_images"

# New dataset
DESTINATION = r"D:\smartcell\dataset"

# Classes
CLASSES = ["Parasitized", "Uninfected"]

# Reproducible random split
random.seed(42)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


for class_name in CLASSES:

    source_folder = os.path.join(SOURCE, class_name)

    # Get image files
    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    # Shuffle images
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    print(f"\n{class_name}")
    print("Total:", total)
    print("Train:", len(train_images))
    print("Validation:", len(val_images))
    print("Test:", len(test_images))

    # Copy images
    for split_name, split_images in splits.items():

        destination_folder = os.path.join(
            DESTINATION,
            split_name,
            class_name
        )

        os.makedirs(destination_folder, exist_ok=True)

        for image in split_images:

            source_path = os.path.join(
                source_folder,
                image
            )

            destination_path = os.path.join(
                destination_folder,
                image
            )

            shutil.copy2(source_path, destination_path)


print("\n================================")
print("DATASET SPLIT COMPLETED")
print("================================")