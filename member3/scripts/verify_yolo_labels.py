"""
verify_yolo_labels.py
---------------------
Visual verification script for SmartCell YOLO dataset.
Draws YOLO bounding boxes on microscope images from D:\smartcell\yolo_malaria.

Generates:
1. Full-frame overview previews with class legends.
2. Full-resolution zoomed-in ROI crops focusing on Parasitized RBCs and WBCs
   to visually verify that bounding box coordinates align with cell boundaries.

Outputs saved to:
D:\smartcell\results\yolo_preview
"""

import os
import cv2
import numpy as np

YOLO_DIR = r"D:\smartcell\yolo_malaria"
PREVIEW_DIR = r"D:\smartcell\results\yolo_preview"

# Class configurations
CLASS_INFO = {
    0: {"name": "Uninfected RBC", "color": (46, 204, 113)},   # Green (BGR: Emerald)
    1: {"name": "Parasitized RBC", "color": (46, 46, 231)},   # Red/Crimson
    2: {"name": "WBC", "color": (230, 126, 34)}               # Royal Blue/Amber
}


def load_yolo_labels(label_path, img_w, img_h):
    """Loads normalized YOLO labels and converts them back to pixel coordinates."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            xc = float(parts[1]) * img_w
            yc = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h

            xmin = int(round(xc - bw / 2.0))
            ymin = int(round(yc - bh / 2.0))
            xmax = int(round(xc + bw / 2.0))
            ymax = int(round(yc + bh / 2.0))

            boxes.append((cls_id, xmin, ymin, xmax, ymax, bw, bh))
    return boxes


def draw_legend(img, counts):
    """Draws an informative summary overlay banner at the top of the image."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (20, 20), (600, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (20, 20), (600, 150), (200, 200, 200), 1)

    cv2.putText(img, "SmartCell YOLO Annotation Preview", (35, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    y_pos = 80
    for cls_id in [0, 1, 2]:
        name = CLASS_INFO[cls_id]["name"]
        color = CLASS_INFO[cls_id]["color"]
        cnt = counts.get(cls_id, 0)
        cv2.rectangle(img, (35, y_pos - 14), (55, y_pos + 4), color, -1)
        cv2.putText(img, f"{name}: {cnt}", (65, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        y_pos += 26


def main():
    print("=" * 70)
    print("SmartCell: YOLO Bounding Box Visual Verification")
    print("=" * 70)

    os.makedirs(PREVIEW_DIR, exist_ok=True)

    # Find sample images across splits with interesting classes
    selected_samples = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(YOLO_DIR, "images", split)
        lbl_dir = os.path.join(YOLO_DIR, "labels", split)

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])

        # Score files by presence of parasitized cells and WBCs
        candidates = []
        for lf in lbl_files:
            stem = os.path.splitext(lf)[0]
            img_file = f"{stem}.jpg"
            img_path = os.path.join(img_dir, img_file)
            if not os.path.exists(img_path):
                continue

            with open(os.path.join(lbl_dir, lf)) as f:
                lines = f.readlines()
            classes = [int(l.split()[0]) for l in lines if l.strip()]
            par_count = sum(1 for c in classes if c == 1)
            wbc_count = sum(1 for c in classes if c == 2)
            candidates.append((split, img_file, lf, par_count, wbc_count, len(classes)))

        # Pick top 2 most informative images from each split
        candidates.sort(key=lambda x: (x[3] > 0, x[4] > 0, x[3] + x[4]), reverse=True)
        for cand in candidates[:2]:
            selected_samples.append(cand)

    print(f"Selected {len(selected_samples)} sample images for visual inspection:")
    for split, img_f, lbl_f, par_cnt, wbc_cnt, total_cnt in selected_samples:
        print(f"  [{split.upper():5}] {img_f} -> Total: {total_cnt:3}, Par: {par_cnt:2}, WBC: {wbc_cnt:1}")

    generated_previews = []

    for split, img_file, lbl_file, par_cnt, wbc_cnt, total_cnt in selected_samples:
        stem = os.path.splitext(img_file)[0]
        img_path = os.path.join(YOLO_DIR, "images", split, img_file)
        lbl_path = os.path.join(YOLO_DIR, "labels", split, lbl_file)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading {img_path}")
            continue

        h, w = img.shape[:2]
        boxes = load_yolo_labels(lbl_path, w, h)

        counts = {0: 0, 1: 0, 2: 0}
        for b in boxes:
            counts[b[0]] += 1

        # 1. Full-frame annotated overview (high quality)
        full_ann = img.copy()
        # Draw Uninfected RBCs with thinner line
        for cls_id, xmin, ymin, xmax, ymax, _, _ in boxes:
            if cls_id == 0:
                cv2.rectangle(full_ann, (xmin, ymin), (xmax, ymax), CLASS_INFO[0]["color"], 2)

        # Draw WBCs and Parasitized cells on top with thicker lines
        for cls_id, xmin, ymin, xmax, ymax, _, _ in boxes:
            if cls_id in [1, 2]:
                color = CLASS_INFO[cls_id]["color"]
                cv2.rectangle(full_ann, (xmin, ymin), (xmax, ymax), color, 4)
                # Label badge
                badge = CLASS_INFO[cls_id]["name"]
                cv2.putText(full_ann, badge, (xmin, max(30, ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

        # Draw legend on full frame
        draw_legend(full_ann, counts)

        # Save resized overview (1920px wide) for easy browsing
        overview_h = int(1920 * h / w)
        overview_img = cv2.resize(full_ann, (1920, overview_h), interpolation=cv2.INTER_AREA)
        overview_out = os.path.join(PREVIEW_DIR, f"overview_{split}_{stem}.jpg")
        cv2.imwrite(overview_out, overview_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        generated_previews.append(overview_out)
        print(f"Generated overview: {overview_out}")

        # 2. High-resolution 800x800 crops around Parasitized RBCs and WBCs
        target_boxes = [b for b in boxes if b[0] in [1, 2]]
        crop_idx = 0
        for cls_id, xmin, ymin, xmax, ymax, bw, bh in target_boxes[:3]:
            crop_idx += 1
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            crop_sz = 800
            half = crop_sz // 2

            c_xmin = max(0, cx - half)
            c_ymin = max(0, cy - half)
            c_xmax = min(w, c_xmin + crop_sz)
            c_ymax = min(h, c_ymin + crop_sz)

            # Re-adjust start if near right/bottom border
            c_xmin = max(0, c_xmax - crop_sz)
            c_ymin = max(0, c_ymax - crop_sz)

            crop_raw = img[c_ymin:c_ymax, c_xmin:c_xmax].copy()

            # Find all boxes that overlap this crop region
            for b_cls, b_x1, b_y1, b_x2, b_y2, _, _ in boxes:
                # Check overlap
                if not (b_x2 < c_xmin or b_x1 > c_xmax or b_y2 < c_ymin or b_y1 > c_ymax):
                    rel_x1 = max(0, b_x1 - c_xmin)
                    rel_y1 = max(0, b_y1 - c_ymin)
                    rel_x2 = min(crop_sz, b_x2 - c_xmin)
                    rel_y2 = min(crop_sz, b_y2 - c_ymin)

                    color = CLASS_INFO[b_cls]["color"]
                    thick = 3 if b_cls in [1, 2] else 2
                    cv2.rectangle(crop_raw, (rel_x1, rel_y1), (rel_x2, rel_y2), color, thick)

                    if b_cls in [1, 2]:
                        badge = CLASS_INFO[b_cls]["name"]
                        cv2.putText(crop_raw, badge, (rel_x1, max(25, rel_y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            crop_name = f"crop_{split}_{stem}_{CLASS_INFO[cls_id]['name'].replace(' ', '_')}_{crop_idx}.jpg"
            crop_out = os.path.join(PREVIEW_DIR, crop_name)
            cv2.imwrite(crop_out, crop_raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
            generated_previews.append(crop_out)
            print(f"Generated zoom crop: {crop_out}")

    print("\n" + "=" * 70)
    print(f"Visual verification complete! {len(generated_previews)} preview images generated in:")
    print(f"{PREVIEW_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
