"""Apply phone-camera colour processing to raw renders: contrast, then saturation.

Reads render_path and writes path for every row of labels.csv, using the per-image
cam_contrast and cam_saturation sampled by make_scene_params.py. Narrow has both at 1.0, so its
renders are copied unchanged (no colour round trip, no second JPEG compression).
Finished images are skipped, so it can be re-run after an interrupted batch.

    uv run python src/camera_process.py --labels data/synthetic/wide/labels.csv
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

JPEG_QUALITY = 95  # same as the renders (D7)


def camera_process(image, contrast, saturation):
    # Contrast first, as a camera's tone curve comes before its colour boost
    image = ImageEnhance.Contrast(image).enhance(contrast)
    # Saturation scales only the HSV S channel, so it cannot change brightness (V).
    # ImageEnhance.Color blends toward grey, which also lowered V for orange fruit and
    # cut defective value coverage from 84% to 75% in pilot 5.
    hsv = np.asarray(image.convert("HSV")).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    return Image.fromarray(hsv.round().astype(np.uint8), mode="HSV").convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/synthetic/wide/labels.csv")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    done = skipped = missing = 0
    for row in labels.itertuples():
        out_path = Path(row.path)
        if out_path.exists():
            skipped += 1
            continue
        if not Path(row.render_path).exists():
            missing += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Temporary name, renamed when complete, so an interrupted run leaves no truncated image
        part_path = out_path.with_suffix(".part.jpg")
        if row.cam_contrast == 1.0 and row.cam_saturation == 1.0:
            shutil.copyfile(row.render_path, part_path)
        else:
            image = Image.open(row.render_path).convert("RGB")
            camera_process(image, row.cam_contrast, row.cam_saturation).save(part_path, quality=JPEG_QUALITY)
        os.replace(part_path, out_path)
        done += 1
    print(f"Processed {done}, skipped {skipped} already done, {missing} renders missing")


if __name__ == "__main__":
    main()
