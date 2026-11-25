# Spec: Stage 3 sim-to-real (Steps 3.1 to 3.4)

Status: decisions S1 to S7 accepted by AJ on 2026-11-25, plus the mixed-training definition (section 7) and the hypotheses (section 9). Nothing in this file is built yet.

## 1. Goal

Measure how well a ResNet18 trained only on Blender renders grades real oranges, compare narrow and wide randomisation (D4b), and see how much a little real data closes the gap. The real test set is scored once per finished run, only by `src/evaluate.py`.

## 2. What we start from

- **Synthetic data (Stage 2, v2 batch):** `data/synthetic/narrow_v2` and `data/synthetic/wide_v2`, 3,000 images each (1,000 per grade, defect types balanced 333/333/334). Splits 2,550 train / 450 val, stratified by grade and defect type. Wide keeps phash duplicate groups in one split; narrow is split without groups (spec_blender section 9).
- **Real data (Stage 1):** train 6,300, val 1,350, test 1,350 (450 per grade; defective is 150 each of blackspotted, cankered, rotten).
- **Upper bound:** `real_lrb1e-4`, test accuracy 0.9985 (2 errors, both fresh called defective). Per source class: fresh 0.9956, everything else 1.0. It is an upper bound, not a fair target: the same physical fruit appears in train, val and test (decisions.md, Stage 1). So the measured gap overstates the real one.
- **Code to reuse:** `dataset.py` (OrangeDataset reads `path` and `grade`, which the synthetic split CSVs also have), `model.py` (ResNet18, MPS device), `train.py` (two phases: head only at 1e-3, then all layers at 1e-4), `evaluate.py` (test metrics, per-source-class accuracy, confusion matrix, Grad-CAM, the run-once guard).

## 3. Step 3.1: Train on synthetic only

- `uv run python src/train.py --train-source synthetic --synthetic-dir data/synthetic/wide_v2 --run-id synth_wide_s42`
- Same recipe as Stage 1: ResNet18 ImageNet weights, phase A head only (lr 1e-3), phase B all layers (lr 1e-4), Adam, batch 32, seed 42, same train transform (section 6). Epoch count and checkpoint rule from S1 and S2.
- Each epoch reports three numbers: train accuracy, synthetic val accuracy, and real val accuracy. The curves figure gets a third line for real val. Real val is for watching only, under the rule in section 5.
- Then `uv run python src/evaluate.py --checkpoint checkpoints/synth_wide_s42.pt --notes "..."` once. The notes state the checkpoint rule (the spec_blender section 10 note).

## 4. Step 3.2: Narrow vs wide (D4b)

- Two runs that differ only in `--synthetic-dir`. Trained identically means: same train set size (2,550), same split rule (85/15, stratified by grade and defect type), same seed, transforms, epochs, learning rates, batch size and checkpoint rule.
- What differs is only the rendered data: narrow is one light background, one top-down camera, one light, tight real-matched colours, no camera processing. Wide varies background, camera, light, colour and camera processing.
- Confound to state in the README: narrow is matched to the look of the real photos (light background), so narrow vs wide is "matched but narrow" against "varied but less matched", not pure variety against none.
- One seed gives no error bar. On 1,350 test images, at 80% accuracy the binomial standard error is about 1.1 points, and training noise adds more. S5 proposes 3 seeds for these two settings.

## 5. Checkpoint selection

Synthetic val is for watching curves only. Narrow val is near-duplicate with narrow train (one framing, one background), so it will sit near 100% and say nothing. More generally, synthetic val accuracy measures the synthetic task, not the real one, and that difference is exactly the gap being measured.

| Option | How | Pros | Cons |
|---|---|---|---|
| (a) Best epoch on real val | Save the epoch with the highest real val accuracy | Picks the epoch that transfers best; common in sim-to-real papers | Uses real labels, so the run is no longer synthetic-only. Real val shares physical fruit with real test, so it is close to choosing on test. Gives an optimistic number |
| (b) Fixed epochs, keep the last | Epoch count fixed before any Stage 3 run; save the final weights | No real labels in any decision, honest synthetic-only number, simple to explain | The last epoch may not be the best for real data; if synthetic training overfits the synthetic look, real accuracy can fall in later epochs |
| (c) Best epoch on synthetic val | As Stage 1 | Uses only synthetic data | Narrow val saturates early, so the choice is close to arbitrary; it optimises the wrong target |

Recommendation: (b), with the epoch count fixed in this spec (S2). Real val accuracy is still computed every epoch and plotted, so the report can show how real accuracy moves during synthetic training. The rule: epoch counts are not changed after real val curves have been seen. Any later change counts as tuning on real val and is logged as such.

The experiments.csv `val_acc` column holds real val accuracy at the saved checkpoint for every Stage 3 run, because it is the only val set shared by all runs (synthetic val differs between narrow and wide). The notes say "fixed epochs; val_acc is real val, not used for selection".

## 6. Augmentation and colour jitter

- The Stage 1 train transform already has horizontal and vertical flips plus brightness, contrast and saturation jitter of 0.2, with no hue jitter (hue separates green from fresh).
- The spec_blender section 10 note asks for mild saturation and contrast jitter on synthetic images to cover very saturated real pixels. The existing 0.2 saturation jitter (x0.8 to x1.2) already does that, on top of wide's camera processing.
- Proposal: use the same transform for every train source (real, synthetic, mixed), so the only thing that changes between runs is the data. S3 lists the alternatives.
- Effect on D4b: the jitter is applied to narrow too, so narrow training images are a little less narrow than the renders. Both settings get the same jitter, so the comparison is still about the data.

## 7. Step 3.3: Mixed training

**Definition: pretrain on wide synthetic, then fine-tune on a real subset.** The mixed run starts from the finished `synth_wide_s42` checkpoint from 3.1 and trains on the real subset only, with the same two-phase recipe, epochs and checkpoint rule (S1, S2). The 3-grade head is kept, not re-initialised, because it already maps to the same grades.

Why sequential, not joint training (synthetic and real shuffled into one set):
- It pairs exactly with the real-only control (S6). Both runs see the same real images with the same recipe; the only difference is the starting weights: ImageNet for real-only, ImageNet then synthetic for mixed. So any difference is what the synthetic pretraining adds.
- It reuses the 3.1 checkpoint, so each mixed run is as cheap as a real-only run.
- It matches how a synthetic-first grading camera would be used in practice: train on renders, then adapt with a small labelled batch from the line.
- Joint training would need a choice of mixing ratio or reweighting (2,550 synthetic against 315 real at 5%), which is one more knob. I see no strong reason to prefer it here. Its main advantage, that the model cannot forget the synthetic variety, is something the sequential runs can check: synthetic val accuracy is still plotted during fine-tuning.

**Real subsets:**
- Drawn from real train by phash group (`group_id` from Stage 1, 5,847 groups in 6,300 images), stratified by source class, seed 42. Method: StratifiedGroupKFold with 20 folds over real train, as in Stage 1. 5% = fold 0, 10% = folds 0 to 1, 25% = folds 0 to 4. That gives about 315, 630 and 1,575 images, nested, each larger subset adding to the smaller one.
- The same subset is used for the real-only run and the mixed run at each fraction, saved as `data/real/splits/train_<pct>.csv` so both runs read the same file.

**Output:** `results/real_fraction.png`. Real test accuracy against real fraction, with a mixed line and a real-only line, the synthetic-only wide mean at 0% (with its seed range), and the 0.9985 upper bound as a dashed line.

**Caveat for the README:** real train shares physical fruit with real test, so any real fraction partly teaches the model the test fruit. Real-only and mixed use the same subset, so they share that overlap equally. The comparison between them is fair, but both absolute numbers are optimistic.

## 8. Evaluation and reporting

**Class balance:** the real test set is 450 fresh, 450 green and 450 defective (checked 2026-11-25), so accuracy equals balanced accuracy over grades and no separate balanced metric is needed. Within defective the source classes are 150 each, and their numbers are per-class accuracy (recall), which does not depend on balance. Macro-F1 is printed anyway in the per-grade report.

**Per run (evaluate.py, unchanged except the notes):** test accuracy, per-grade precision/recall/F1, per-source-class accuracy (blackspotted, cankered, rotten, fresh, green), confusion matrix, Grad-CAM grid on 2 test images per grade plus up to 4 errors, and a predictions CSV.

**Across runs (new `src/compare_runs.py`, reads only experiments.csv and the saved predictions CSVs, never the test images):**
- `results/stage3_summary.csv`: one row per run with test accuracy, recall per grade and accuracy per source class, next to `real_lrb1e-4`. The gap column is 0.9985 minus test accuracy.
- The 3.3 plot above.
- With 3 seeds (S5): mean and min to max per setting.

**Grad-CAM comparison (S7):** one figure, `results/gradcam_compare.png`. Rows are the same fixed images, columns are Stage 1 real, synthetic narrow, synthetic wide and the best mixed run. Each cell shows the heatmap for that model's own prediction. The question it answers: do synthetic-trained models look at the fruit and its marks the way the real model does, or at edges, background or the stem? Stage 1's heatmaps sat on the fruit; a synthetic model looking at the background or the frame edge would point at a specific rendering cue.

## 9. Hypotheses (written before any Stage 3 run)

Expected outcomes. They are written down now so the results can be read against them, not explained after the fact. None of them is a measured number.

- **H1, synthetic-only transfers but with a clear gap.** Synthetic-only wide is well above chance (33%) and well below the upper bound. Most errors are defective against fresh: defect appearance is the least realistic part of the renders, and defective brightness is the largest colour gap (spec_blender section 9). Green is the easiest grade, because hue separates it and green colour coverage is 99 to 100%.
- **H2, wide beats narrow.** Domain randomisation should transfer better than one matched look. It is a real possibility that narrow does as well or better, because narrow is styled like the real photos. If so, that is the result, and it says variety did not help on this test set.
- **H3, defect types differ.** Rot (large patches) is detected most reliably. Canker is the weakest, because the synthetic canker marks are the roughest approximation. Blackspot is in between.
- **H4, synthetic pretraining helps most when real data is scarce.** Mixed beats real-only at 5%, and the advantage shrinks by 25%. Because of the same-fruit leak, both may already be near the upper bound at 10 to 25%, and then no difference shows.
- **H5, attention.** Synthetic-trained models still look at the fruit (Grad-CAM), but more at the outline and at dark regions than the real model does. Heatmaps on the background or the frame edge would point at a rendering cue.

**What counts as a real difference:**
- **Narrow vs wide (3 seeds each):** the difference in mean test accuracy must be more than 2x the larger seed range (max minus min) of the two settings, and more than 2 points. The 2-point floor is about 2 binomial standard errors of a test accuracy near 80% on 1,350 images, so a gap smaller than test sampling noise never counts, even if the seeds agree closely.
- **Mixed vs real-only (one seed each):** there is no seed spread for these runs, so the training noise estimate is the wide synthetic-only seed range. The difference must be more than 2x that range and more than 2 points.
- **Near the ceiling:** above about 97%, differences come down to a handful of images. There I report error counts (for example 4 against 9 errors out of 1,350) and do not call a difference real unless it passes both rules above.
- Anything that fails these rules is reported as "no clear difference", with both numbers.

## 10. Open decisions (accepted 2026-11-25)

**S1. Checkpoint rule.** (a) best real val, (b) fixed epochs, keep last, (c) best synthetic val. Recommend (b), as in section 5.

**S2. Epoch count.** (a) 5+5, the same as Stage 1. (b) 10+10, so the number of weight updates is closer to Stage 1 (2,550 images per epoch against 6,300). (c) Pick from synthetic val curves of a trial run. Recommend (a): the identical recipe is easiest to defend, and more epochs on synthetic data risk fitting the synthetic look more closely. Escape hatch, a fixed rule set on 2026-11-25 before any Stage 3 run: phase A counts as unconverged only if the synthetic train loss at epoch 5 is more than 10% below the loss at epoch 4, in the `synth_wide_s42` log. It is applied mechanically from the train loss column only, before real val is discussed. Whatever it gives (5+5 or 10+10) applies to every synthetic-only and mixed run, not just this one.

**S3. Colour jitter.** (a) Same 0.2 brightness, contrast and saturation jitter as Stage 1, for every source. (b) Stronger saturation and contrast jitter (0.3 to 0.4) for synthetic images only. (c) No jitter for synthetic. Recommend (a): one transform for all runs keeps the comparison clean, and 0.2 already covers the gap the spec note describes. (b) would make the train transform depend on the data source, which is one more thing that differs between runs.

**S4. Synthetic setting for 3.3 (the pretraining checkpoint).** (a) Wide, fixed now. (b) Whichever of narrow and wide does better in 3.2. (c) Both. Recommend (a): wide is the setting the project is built around. (b) uses test results to make a design choice, which is tuning on test. (c) is honest but adds 3 runs and a second line to every plot. Mixed starts from the seed 42 wide checkpoint.

**S5. Seeds.** (a) One seed (42) for every run. (b) 3 seeds (42, 43, 44) for the 3.1/3.2 synthetic-only runs, one seed for 3.3. Recommend (b): narrow vs wide is the headline comparison, and one seed cannot tell a 2-point difference from noise. It costs 4 extra runs (about 25 min). Same rendered data for every seed, so the spread is training noise only. MPS is not fully deterministic, so even the same seed can differ slightly between reruns.

**S6. Real-only controls for 3.3.** (a) Mixed runs only, as written in CLAUDE.md. (b) Add real-only runs at 5%, 10% and 25%, on the same subsets as the mixed runs. Recommend (b): Stage 1 reached 0.98 real val after phase A alone, so a few hundred real images may already score near the upper bound. Without the control, a high mixed score cannot be credited to the synthetic data. It costs 3 short runs (about 10 min).

**S7. Images for the Grad-CAM comparison.** (a) Fixed real val images. (b) Real test images. Recommend (a): 1 image per source class plus 1 more fresh and green (7 images), chosen by seed before any Stage 3 run. That keeps the test set read only by evaluate.py, and each run's evaluate.py still writes its own Grad-CAM on test images.

## 11. Estimated training time (M1, 8 GB)

From Stage 1 (6,300 train images, 1,350 val): phase A about 40 s per epoch, phase B about 95 s (up to 122 s when the Mac got warm). That is about 210 images per second for head-only training, 75 for full training, and 150 for evaluation, including JPEG decoding.

| Run | Train images | Per epoch (A / B), incl. synthetic and real val | 5+5 total |
|---|---|---|---|
| Synthetic only (narrow or wide) | 2,550 | ~25 s / ~50 s | ~6 min |
| Mixed 5% / 10% / 25% (fine-tune from synth_wide_s42) | 315 / 630 / 1,575 | ~12-17 s / ~15-30 s | ~2-4 min |
| Real-only 5% / 10% / 25% (S6) | 315 / 630 / 1,575 | ~12-17 s / ~15-30 s | ~2-4 min |
| evaluate.py per run | 1,350 test | | ~1 min |

Plan: 6 synthetic-only runs (3 seeds x 2 settings) at about 6 min, 3 mixed and 3 real-only at about 3 min each, plus 12 evaluations at about 1 min. About 1 h 5 min of compute. Mixed needs no extra pretraining because it reuses synth_wide_s42. With 10+10 epochs (S2b), roughly double the training part. Memory is not a concern: batch 32 at 224 px ran in Stage 1.

## 12. Code changes (for the plan of each step)

- `train.py`: `--train-source real|synthetic|mixed`, `--synthetic-dir`, `--real-fraction`, and `--init-checkpoint` for mixed (start from synth_wide_s42 instead of ImageNet). Evaluate real val each epoch alongside the run's own val; mixed runs also track wide synthetic val to show any forgetting. Save the checkpoint by the S1 rule. The real path stays as it is, so Stage 1 reproduces. The stray blank line already in the working copy goes in with this change.
- `evaluate.py`: `real_fraction` 0.0 for synthetic, the fraction for mixed; `val_acc` is real val at the checkpoint. Otherwise unchanged.
- `src/make_real_subsets.py` (new, small): writes `data/real/splits/train_{5,10,25}.csv` from real train by phash group, stratified by source class, nested.
- `src/compare_runs.py` (new): summary CSV, real-fraction plot, Grad-CAM comparison on real val images.
- Run IDs: `synth_<setting>_s<seed>`, `mixed_wide_<pct>`, `real_<pct>`.

## 13. Honest limits

- The upper bound is inflated by the same-fruit leak, so the gap is overstated.
- The real 5/10/25% subsets come from real train, which shares physical fruit with real test. Real-only and mixed runs use the same subsets, so they carry the overlap equally: the comparison between them is fair, but both absolute accuracies are optimistic.
- The real test set is one dataset from one source: white or grey backgrounds and phone photos. Good transfer here does not show transfer to a grading machine's camera.
- With S1(b), the synthetic-only number uses no real labels in any training decision. Colour ranges did come from real train statistics (Stage 2), which the README states.
- One model (ResNet18) and one recipe. No hyperparameter search for the synthetic runs.

## 14. Out of scope for Stage 3

- No re-rendering or generator changes. If results point at a rendering cue, it goes in the README as a next step.
- No domain adaptation methods (feature alignment, style transfer), no other architectures.
- Step 3.4 (README) uses only numbers from logged runs.
