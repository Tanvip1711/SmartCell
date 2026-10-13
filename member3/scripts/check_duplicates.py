import os
import hashlib

base = r"D:\smartcell\cell_images"

classes = ["Parasitized", "Uninfected"]

hashes = {}
duplicates = []

for class_name in classes:

    folder = os.path.join(base, class_name)

    print(f"\nScanning: {class_name}")

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        filepath = os.path.join(folder, filename)

        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in hashes:
            duplicates.append(
                (filepath, hashes[file_hash])
            )
        else:
            hashes[file_hash] = filepath

print("\n==============================")
print("DUPLICATE CHECK RESULT")
print("==============================")

print("Total unique images :", len(hashes))
print("Duplicate images    :", len(duplicates))

if duplicates:
    print("\nDuplicate files found:")

    for duplicate, original in duplicates[:20]:
        print("\nDuplicate :", duplicate)
        print("Original  :", original)
else:
    print("\nNo exact duplicate images found.")