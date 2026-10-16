from rembg import remove, new_session
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────
INPUT_ROOT  = "D:\\horse4\\dataset\\horse"
OUTPUT_ROOT = "D:\\horse4\\dataset\\horse_seg_v3"  # NEW folder — horse_seg stays safe
# ─────────────────────────────────────────────────────────

# u2netp = faster model (recommended for 29k images)
# u2net  = slower but more accurate (swap if you have time)
print("Loading rembg model (downloads ~5MB on first run)...")
session = new_session("u2netp")
print("Model loaded. Starting segmentation...\n")

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
            out_path = os.path.join(out_dir, img_file)

            try:
                # Open image with PIL (rembg requires PIL)
                pil_img = Image.open(img_path).convert("RGB")

                # Remove background — returns RGBA image
                output = remove(pil_img, session=session)

                # Convert RGBA to numpy
                output_np = np.array(output)  # shape: H x W x 4

                # Extract alpha channel as mask
                alpha = output_np[:, :, 3]     # 0=background, 255=foreground
                mask  = (alpha > 128).astype(np.uint8)

                # Read original with OpenCV for BGR output
                img_bgr = cv2.imread(img_path)

                # Apply mask — background becomes black
                segmented = img_bgr * mask[:, :, np.newaxis]

                # Check if mask found anything meaningful
                coverage = mask.mean()  # 0.0 to 1.0
                if coverage > 0.05:     # at least 5% of pixels are foreground
                    cv2.imwrite(out_path, segmented)
                    detected += 1
                else:
                    # Mask too small — save original as fallback
                    cv2.imwrite(out_path, img_bgr)
                    fallback += 1

            except Exception as e:
                # Any error — save original as fallback
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    cv2.imwrite(out_path, img_bgr)
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
