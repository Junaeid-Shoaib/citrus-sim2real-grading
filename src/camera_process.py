"""Apply phone-camera colour processing to raw renders: contrast, then saturation.

Reads render_path and writes path for every row of labels.csv, using the per-image
cam_contrast and cam_saturation sampled by make_scene_params.py (1.0 in narrow, so a no-op).
Finished images are skipped, so it can be re-run after an interrupted batch.

    uv run python src/camera_process.py --labels data/synthetic/wide/labels.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance

JPEG_QUALITY = 95  # same as the renders (D7)


def camera_process(image, contrast, saturation):
    # Contrast first, as a camera's tone curve comes before its colour boost. ImageEnhance.Color
    # above 1 pushes colours away from grey and clips at 255, like an over-saturated phone photo.
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageEnhance.Color(image).enhance(saturation)


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
        image = Image.open(row.render_path).convert("RGB")
        camera_process(image, row.cam_contrast, row.cam_saturation).save(out_path, quality=JPEG_QUALITY)
        done += 1
    print(f"Processed {done}, skipped {skipped} already done, {missing} renders missing")


if __name__ == "__main__":
    main()
