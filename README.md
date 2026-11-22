# citrus-sim2real-grading

Grading oranges as fresh, green or defective with a CNN, and testing whether a model trained only on synthetic Blender renders works on real photos.

## Goal

Fruit-grading cameras normally need thousands of hand-labelled photos. This project tests whether synthetic images rendered in Blender, where the label comes free from the render parameters, can replace that labelling. The headline result is the sim-to-real gap: accuracy on real fruit for a model trained only on synthetic images, compared with a baseline trained on real images, and what closes the gap.

## Results

Every number here comes from a logged run in `results/experiments.csv`. Accuracy is on the real test set (1,350 images, 450 per grade).

| Run | Trained on | Val acc | Real test acc |
|---|---|---|---|
| real_lrb1e-4 | real | 0.9993 | 0.9985 |

The real baseline is an upper bound, not a fair target: the same physical fruit appears across splits (see Limitations). The only errors were 2 fresh oranges predicted as defective. Grad-CAM heatmaps (`results/real_lrb1e-4_gradcam.png`) sit on the fruit rather than the background.

## Limitations

- Blender rigid-body physics is used only for natural resting poses. It is not DEM and does not model fruit-to-fruit contact or bulk flow.
- The sorting loop in Stage 4 is a simulation of a robot interface, not robot code.
- The real-data baseline is an upper bound. The split groups augmented copies of the same photo together, but the dataset also contains the same physical orange photographed several times, and those photos can land in different splits. Spot checks found val images whose closest train image is clearly the same fruit. So the real baseline partly measures recognising seen fruit, and the sim-to-real gap is likely overstated. Testing on an independent orange dataset would give an honest number and is left as future work.

## Data

Real images: *Orange Fruit Image Dataset for Classification*, Mendeley Data, DOI [10.17632/zk4hzhtcdd.2](https://doi.org/10.17632/zk4hzhtcdd.2). Licensed under CC BY 4.0.

## Setup

```
uv sync
uv run python src/<script>.py
```
