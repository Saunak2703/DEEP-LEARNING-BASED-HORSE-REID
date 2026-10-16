from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────
INPUT_ROOT  = "D:\\horse4\\dataset\\horse"
OUTPUT_ROOT = "D:\\horse4\\dataset\\horse_seg"
# ─────────────────────────────────────────────────────────

model = YOLO("yolov8m-seg.pt")  # downloads ~52MB on first run only

total_images   = 0
total_detected = 0
total_fallback = 0

for split in ["train", "test"]:
    split_path = os.path.join(INPUT_ROOT, split)
    if not os.path.exists(split_path):
        continue

    for horse_folder in sorted(os.listdir(split_path)):
        in_dir = os.path.join(split_path, horse_folder)
        if not os.path.isdir(in_dir):
            continue

        out_dir = os.path.join(OUTPUT_ROOT, split, horse_folder)
        os.makedirs(out_dir, exist_ok=True)

        files = [f for f in os.listdir(in_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        detected = 0
        fallback = 0

        for img_file in tqdm(files, desc=f"{split}/{horse_folder}"):
            img_path = os.path.join(in_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            out_path   = os.path.join(out_dir, img_file)
            mask_found = False

            # Cascade: try conf=0.3 first, then 0.15
            # Lower confidence = catches more horses
            for conf in [0.3, 0.15]:
                results = model(img, classes=[17], conf=conf, verbose=False)[0]

                if results.masks is not None:
                    # Pick LARGEST mask = main horse in the image
                    masks = results.masks.data.cpu().numpy()
                    areas = [m.sum() for m in masks]
                    best_mask = masks[int(np.argmax(areas))]

                    # Resize mask to image dimensions
                    mask = cv2.resize(best_mask,
                                      (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                    mask = (mask > 0.5).astype(np.uint8)

                    # Background becomes black, horse body kept
                    # OVERWRITES existing file if already exists
                    segmented = img * mask[:, :, np.newaxis]
                    cv2.imwrite(out_path, segmented)

                    detected   += 1
                    mask_found  = True
                    break

            if not mask_found:
                # Save original as fallback — OVERWRITES existing file
                cv2.imwrite(out_path, img)
                fallback += 1

        total_images   += len(files)
        total_detected += detected
        total_fallback += fallback

        pct = int(100 * detected / max(len(files), 1))
        print(f"  {split}/{horse_folder}: "
              f"{detected}/{len(files)} segmented ({pct}%)  |  "
              f"{fallback} kept original")

# Final summary
print("\n" + "=" * 55)
print("DONE")
print(f"Total images processed  : {total_images}")
pct_det = int(100 * total_detected / max(total_images, 1))
pct_fb  = int(100 * total_fallback / max(total_images, 1))
print(f"Successfully segmented  : {total_detected}  ({pct_det}%)")
print(f"Kept original (fallback): {total_fallback}  ({pct_fb}%)")
print(f"Output saved to         : {OUTPUT_ROOT}")
print("=" * 55)
