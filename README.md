# citrus-sim2real-grading

Grading oranges as fresh, green or defective with a CNN, and testing whether a model trained only on synthetic Blender renders works on real photos.

## Goal

Fruit-grading cameras normally need thousands of hand-labelled photos. This project tests whether synthetic images rendered in Blender, where the label comes free from the render parameters, can replace that labelling. The headline result is the sim-to-real gap: accuracy on real fruit for a model trained only on synthetic images, compared with a baseline trained on real images, and what closes the gap.

## Results

No results yet. Every number here will come from a logged run in `results/experiments.csv`.

## Limitations

- Blender rigid-body physics is used only for natural resting poses. It is not DEM and does not model fruit-to-fruit contact or bulk flow.
- The sorting loop in Stage 4 is a simulation of a robot interface, not robot code.

## Data

Real images: *Orange Fruit Image Dataset for Classification*, Mendeley Data, DOI [10.17632/zk4hzhtcdd.2](https://doi.org/10.17632/zk4hzhtcdd.2). Licensed under CC BY 4.0.

## Setup

```
uv sync
uv run python src/<script>.py
```
