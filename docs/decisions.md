# Decisions

One line per decision: what, why, alternative rejected.

- Plain scripts in src/ instead of an installable package: simpler to run with `uv run python src/x.py`; rejected the uv --package layout as unneeded for a single project.
- uv for environment and packages: lockfile makes versions reproducible; rejected pip + requirements.txt.
- 256-bit phash over all 8 flips/rotations, Hamming threshold 24, connected components as duplicate groups: pairs at or under 24 looked like the same fruit on inspection; rejected 64-bit phash because it matched different oranges and chained about 2,000 images into one group.
- StratifiedGroupKFold with 20 folds (3 test, 3 val, 14 train) for the 70/15/15 split, stratified on source_class: balances each defect type across splits (grade balance follows) and keeps duplicate groups whole in one call; rejected a random stratified split because augmented copies would leak across splits.
- Balance defective by sampling 1,000 per defect folder before hashing: gives 3,000 per grade and halves hashing time; rejected keeping all 9,000 defective images because it would make defective 60% of the data.
- Train augmentation is horizontal + vertical flip and brightness/contrast/saturation jitter of 0.2, no hue jitter: top-down fruit has no "up", and hue is the cue separating green from fresh; rejected hue jitter because it could make an image contradict its label.
- Fixed label order GRADES = [defective, fresh, green] in dataset.py: every script imports the same mapping; rejected inferring order from the data because it can change with the CSV.
