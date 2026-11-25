"""Sample render parameters for synthetic oranges and write labels.csv plus train/val splits.

All randomness for the synthetic data lives here (D1). blender/generate.py only turns a
parameter row into an image, so each image's grade is fixed before it is rendered.

    uv run python src/make_scene_params.py --randomisation wide --n-per-grade 1000
"""

import argparse
import colorsys
import json
import math
import random
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

GRADES = ["fresh", "green", "defective"]
# Stored as JSON text in labels.csv because they are lists or tuples
JSON_COLUMNS = ["pattern_offset", "rotation", "light_location", "camera_shift", "bg_colour", "patches"]


def load_config(path, randomisation=None):
    with open(path) as f:
        config = yaml.safe_load(f)
    randomisation = randomisation or config["randomisation"]
    # Shared settings (label semantics) plus the narrow or wide block
    return {**config["shared"], **config[randomisation], "randomisation": randomisation}


def draw(rng, spec):
    # A [min, max] list is sampled uniformly; a single number is fixed
    if isinstance(spec, list):
        return rng.uniform(spec[0], spec[1])
    return float(spec)


def draw_int(rng, spec):
    return rng.randint(spec[0], spec[1])


def pick(rng, weights):
    names = list(weights)
    return rng.choices(names, weights=[weights[n] for n in names])[0]


def hsv_to_srgb(rng, ranges):
    return colorsys.hsv_to_rgb(draw(rng, ranges["hue"]), draw(rng, ranges["saturation"]), draw(rng, ranges["value"]))


def sample_defect(rng, cfg, kind):
    params = {"defect_type": kind,
              "shift_dark": draw(rng, cfg["shift_dark"]),
              "shift_dull": draw(rng, cfg["shift_dull"]),
              "shift_rough": draw(rng, cfg["shift_rough"])}
    settings = cfg["defects"][kind]
    params["wrinkle"] = draw(rng, settings.get("wrinkle", 0.0))
    params["canker_density"] = draw(rng, settings.get("density", 0.5))
    if kind in ("spots", "lesions"):
        # Black spots and canker marks share the spot machinery in generate.py
        params["spot_scale"] = draw(rng, settings["scale"])
        params["spot_radius"] = draw(rng, settings["radius"])
        params["spot_distort"] = draw(rng, settings.get("distort", 0.0))
    else:
        # Each rot patch: (degrees from the view direction, azimuth, radius); generate.py places it
        params["patches"] = [(draw(rng, cfg["patch_degrees_from_view"]), rng.uniform(0, 360),
                              draw(rng, settings["radius"]))
                             for _ in range(draw_int(rng, settings["count"]))]
        if rng.random() < cfg["mould_probability"]:
            params["mould_amount"] = draw(rng, cfg["mould_amount"])
            params["mould_mix"] = draw(rng, cfg["mould_mix"])
    return params


def sample_params(grade, rng, cfg, defect_type=None):
    """One image's parameters. `grade` only chooses the colour ranges and whether a defect is added.

    defect_type fixes the defect for a review sheet; for data it is None and drawn evenly.
    """
    kind = "none"
    colours = cfg["grades"][grade]
    if grade == "defective":
        # Each defect type has its own base colour range (canker fruit is yellower than blackspot)
        kind = defect_type or rng.choice(cfg["defect_types"])
        colours = colours[kind]
    shape, light, camera, bg = cfg["shape"], cfg["light"], cfg["camera"], cfg["background"]
    hue = draw(rng, colours["hue"])

    params = {
        "grade": grade,
        "randomisation": cfg["randomisation"],
        "hue": hue,
        "saturation": draw(rng, colours["saturation"]),
        "value": draw(rng, colours["value"]),
        "blotch_hue": hue + draw(rng, colours["blotch_hue_offset"]),
        "blotch_amount": draw(rng, colours["blotch_amount"]),
        "pattern_offset": tuple(draw(rng, cfg["pattern_offset"]) for _ in range(3)),
        "rotation": tuple(draw(rng, cfg["rotation"]) for _ in range(3)),
        "stem_mix": draw(rng, cfg["stem_mix"]),
        "squash": draw(rng, shape["squash"]),
        "lumpiness": draw(rng, shape["lumpiness"]),
        "pit_scale": draw(rng, shape["pit_scale"]),
        "bump_strength": draw(rng, shape["bump_strength"]),
        "roughness": draw(rng, shape["roughness"]),
        # No defect and no shift unless the grade is defective
        "defect_type": "none", "shift_dark": 0.0, "shift_dull": 0.0, "shift_rough": 0.0,
        "spot_scale": 4.0, "spot_radius": 0.03, "spot_distort": 0.0, "wrinkle": 0.0, "canker_density": 0.5, "patches": [],
        "mould_amount": 0.0, "mould_mix": 0.0,
    }
    if grade == "defective":
        params.update(sample_defect(rng, cfg, kind))

    # Light on a hemisphere above the fruit
    elevation, azimuth = math.radians(draw(rng, light["elevation"])), math.radians(draw(rng, light["azimuth"]))
    distance = draw(rng, light["distance"])
    params["light_location"] = (distance * math.cos(elevation) * math.cos(azimuth),
                                distance * math.cos(elevation) * math.sin(azimuth),
                                distance * math.sin(elevation))
    params["light_energy"] = draw(rng, light["energy"])
    params["light_temperature"] = draw(rng, light["temperature"])

    params["camera_tilt"] = draw(rng, camera["tilt"])
    params["camera_azimuth"] = draw(rng, camera["azimuth"])
    params["fill"] = draw(rng, camera["fill"])
    params["camera_shift"] = (draw(rng, camera["shift"]), draw(rng, camera["shift"]))

    bg_type = pick(rng, bg["types"])
    params["bg_type"] = bg_type
    params["bg_colour"] = hsv_to_srgb(rng, bg[bg_type])
    params["bg_pattern"] = pick(rng, bg["patterns"])
    params["bg_pattern_strength"] = draw(rng, bg["pattern_strength"]) if params["bg_pattern"] != "plain" else 0.0
    params["bg_pattern_scale"] = draw(rng, bg["pattern_scale"])
    return params


def make_labels(cfg, n_per_grade, seed, out_dir):
    rows = []
    index = 0
    for grade in GRADES:
        for j in range(n_per_grade):
            # One RNG per image, so any single image can be re-sampled on its own
            image_seed = seed * 1_000_000 + index
            # Defect types cycle so the three are exactly balanced within the defective grade
            kind = cfg["defect_types"][j % len(cfg["defect_types"])] if grade == "defective" else None
            params = sample_params(grade, random.Random(image_seed), cfg, defect_type=kind)
            path = out_dir / grade / f"{grade}_{index:05d}.jpg"
            rows.append({"path": str(path), "seed": image_seed, **params})
            index += 1
    df = pd.DataFrame(rows)
    for column in JSON_COLUMNS:
        df[column] = df[column].map(json.dumps)
    # path, grade and defect_type first, then every sampled parameter
    first = ["path", "grade", "defect_type", "seed"]
    return df[first + [c for c in df.columns if c not in first]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="blender/config.yaml")
    parser.add_argument("--randomisation", choices=["narrow", "wide"])
    parser.add_argument("--n-per-grade", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", help="default: data/synthetic/<randomisation>")
    args = parser.parse_args()

    cfg = load_config(args.config, args.randomisation)
    out_dir = Path(args.out_dir or f"data/synthetic/{cfg['randomisation']}")
    labels = make_labels(cfg, args.n_per_grade, args.seed, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "labels.csv", index=False)

    # 85/15 train/val, stratified by grade. No synthetic test split: the test set is always real.
    train, val = train_test_split(labels, test_size=0.15, stratify=labels["grade"], random_state=args.seed)
    (out_dir / "splits").mkdir(exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        split[["path", "grade", "defect_type"]].to_csv(out_dir / "splits" / f"{name}.csv", index=False)

    print(f"Wrote {len(labels)} rows to {out_dir / 'labels.csv'}")
    print(pd.crosstab(labels["grade"], labels["defect_type"]).to_string())
    print(f"Splits: train {len(train)}, val {len(val)}")


if __name__ == "__main__":
    main()
