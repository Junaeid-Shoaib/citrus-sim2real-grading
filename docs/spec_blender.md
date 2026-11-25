# Spec: Stage 2 synthetic generator (Steps 2.1 to 2.4)

Status: decisions D1 to D8 agreed with AJ on 2026-11-25. Nothing in this file is built yet.

## 1. Goal

Render labelled orange images in Blender so a classifier can be trained without hand labelling. Each image's grade comes from the parameters used to render it, never from looking at the image. Stage 3 trains on these renders and tests once on the real test set.

The renders do not need to be photorealistic. They need to vary enough (colour, peel, defects, light, camera, background) that the real photos look like one more variation. That is domain randomisation, and Step 3.2 tests whether it helps.

## 2. What the real data looks like

From a 20-image sample of the real train split (4 per source class):

- **Background:** almost all white or light grey paper or a light speckled worktop. Soft shadow under the fruit. No dark belts.
- **Framing:** close-up. The fruit fills about 60 to 100% of the frame and is sometimes cropped at the edges. Some images have stretched edge pixels from the authors' augmentation.
- **View:** mostly from the side or at an angle, not straight down. Stems, calyx and navels are often visible.
- **Peel:** strong pebbled texture (oil glands) and specular highlights.
- **Fresh:** saturated orange. **Green:** deep to mid green, sometimes with yellow patches.
- **Defective:**
  - blackspotted: small dark pits on a wrinkled orange peel.
  - cankered: brown or grey raised lesions, often on a greenish-orange fruit.
  - rotten: white or blue-green mould, collapsed or split peel, brown patches.

Three things follow. First, a dark-belt, top-down scene alone will not look like the real data, so backgrounds and camera angle must be randomised (CLAUDE.md already asks for this in 2.3). Second, some defective fruit is greenish, so the defective base colour should not be pure orange or the model may learn "greenish means green grade" (see D5). Third, stems, calyx and navels show up on fresh and green fruit as well as defective, so a small dark mark at the pole is normal peel, not a defect (see D6).

## 3. Measured render cost on this Mac

Benchmark: M1, 8 GB, Blender 5.2.4 LTS, headless (`-b`). Test scene: squashed UV sphere with subdivision and displacement, Principled BSDF, plane, one area light, camera, 224x224. Six frames per engine.

| Engine | First frame | Later frames | Peak memory |
|---|---|---|---|
| EEVEE, 16 samples | 6.5 s (shader compile) | 0.34 s | 350 MB |
| Cycles Metal GPU, 32 samples + denoise | 112 s (kernel compile, cached after first run) | 1.1 to 1.7 s | 420 MB |

EEVEE works in background mode on this Mac, so it is the engine. Cycles stays as the fallback only.

**Chosen settings (fit 8 GB, keep renders fast):**

- Engine EEVEE, 224x224 at 100%, 32 render samples. At 16 samples the shadow edge was visibly grainy.
- Raytracing off, no bloom, no volumetrics, no motion blur.
- Mesh: UV sphere at 64x32 with subdivision level 2. Fine peel pits come from a bump node in the material, not from denser geometry, which keeps memory low.
- Procedural textures only (Noise, Voronoi). No image textures to load.
- One Blender process renders a chunk of images (default 500), then exits. A crash loses at most one chunk, memory is released between chunks, and a re-run skips files that already exist.
- Material node graphs are built once per grade. Each image changes only input values, never the node layout. Changing the layout makes EEVEE recompile shaders (about 6 s each), which would multiply render time by about 15.
- Do not train and render at the same time. Either alone fits in 8 GB; both together will swap.

**Measured time (Step 2.1).** The Step 2.1 scene was timed with 20 renders in one Blender process, changing hue and rotation each time. It took 0.48 s per image on average (max 0.53 s), with a peak memory of 460 MB. Step 2.2 adds defect nodes, so I plan on about 0.6 s per image, plus a few seconds of start-up per chunk. A fanless M1 Air may throttle on long runs and be 20 to 30% slower.

| Images | Estimate at 0.6 s per image |
|---|---|
| 3,000 (1,000 per grade, one setting) | about 30 min |
| 9,000 (3,000 per grade, one setting) | about 1.5 h |
| Both narrow and wide at 1,000 per grade | about 1 h |
| Both narrow and wide at 3,000 per grade | about 3 h |

Renders are deterministic up to GPU float noise. Two runs of the same parameters differed in 42 of 50,176 pixels, by at most 2 of 255 levels.

**Disk:** JPEG at quality 95 is about 20 KB per image (about 400 MB for 18,000). PNG would be about 100 KB.

## 4. Step 2.1: Render one orange

`blender/generate.py`, run as `blender -b -P blender/generate.py -- --out data/synthetic/step2_1_fresh.jpg`. Step 2.4 adds `--labels` (D1).

Scene:
- **Fruit:** UV sphere, radius 1, z-scale about 0.9 (slight squash), subdivision 2. A Displace modifier with a low-frequency noise texture makes the shape slightly lumpy.
- **Peel material:** Principled BSDF.
  - Base colour comes from an HSV value, so hue is one number we can sample.
  - Roughness about 0.4 for the waxy shine.
  - A Voronoi texture drives a Bump node for the oil-gland pitting.
  - A low raised dome at one pole stands in for the stem or calyx (see D6). It is about 4 to 8 px wide at 224 px, its colour is sampled per image from olive green to dark brown (`stem_mix`), and noise mottling and bump make it read as a scar, not a hole.
- **Ground:** plane under the fruit, dark grey belt material by default.
- **Light:** one area light above and to the side.
- **Camera:** looking straight down, framed so the fruit fills about 70% of the image.
- **Output:** one 224x224 JPEG of a fresh orange, fixed seed.

Done when: the image looks like an orange to AJ and the timing per frame is printed.

## 5. Step 2.2: Grades from parameters

A `sample_params(grade, rng, cfg)` function returns a dict of numbers. A `apply_params(params)` function sets them in the scene. The grade is an input to sampling, so the label is known before rendering.

| Grade | Base hue | Extra |
|---|---|---|
| fresh | orange range, e.g. HSV hue 0.06 to 0.10, high saturation | none |
| green | green to yellow-green, e.g. hue 0.16 to 0.30, with optional yellow blotches | none |
| defective | orange, optionally widened toward yellow-green (D5) | one defect type plus a global shift |

Defect types (one per defective image, chosen evenly):
- **spots:** 3 to 30 small dark brown or black spots from a thresholded Voronoi texture, plus a darker bump in the spot. Matches blackspotted.
- **lesions:** fewer, larger brown or grey patches with a yellow halo ring. Matches cankered.
- **rot:** one or two large patches. Inside: brown, very rough, sunken via bump, with whitish or blue-green mould noise on top. Matches rotten.

Global shift for every defective image, sampled not fixed:
- Lower saturation (duller).
- Hue nudged toward brown.
- Higher roughness (less shine).

Each shift is sampled from a range that starts at 0, so some defective fruit has no global shift and is defective only because of its local defect. That stops the model from relying on overall dullness alone.

Stage 1 Grad-CAM showed the real model uses broad peel cues, so local spots alone are not enough.

**Defect placement.** Every defect has a centre point. Centres are sampled after the fruit's random rotation, on the camera-facing hemisphere: each centre lies within about 60 degrees of the direction from the fruit centre to the camera. The centre is then converted into the fruit's own coordinates, so the material puts the defect where it was sampled. This means no defective image has its defect hidden on the far side, which would be a defective label on a clean-looking image.
- spots: no centre; the spot texture is masked to the camera-facing cap (within about 60 degrees of the view direction), so spots spread across the visible face like real blackspot.
- lesions and rot: each patch has its own centre, up to 3.

All three defect types live in one material, switched by mix factors, so changing type does not recompile shaders. The material has 3 fixed centre slots; unused slots get radius 0.

The defect type is saved in labels.csv. The label stays the 3-class grade.

## 6. Step 2.3: Scenario matrix in config.yaml

Every random value is a `[min, max]` range in `config.yaml`. Two blocks, `narrow` and `wide`, selected by `randomisation: narrow|wide`.

| Parameter | narrow | wide |
|---|---|---|
| Fruit hue, saturation, value | grade ranges, tight | grade ranges, wider value and saturation |
| Peel bump strength, pit scale | fixed | range |
| Squash (z-scale), lumpiness | small range | larger range |
| Defect size, count | middle of range | full range |
| Light intensity | fixed | 0.5x to 2x |
| Light colour temperature | 5500 K | 3500 to 7500 K |
| Light position | fixed | random on a hemisphere above |
| Camera | top-down, fixed distance | tilt 0 to 60 degrees, distance so fruit fills 55 to 100% of the frame, small off-centre shift |
| Fruit rotation | random (free, costs nothing) | random |
| Background | light surface only: plain white to light grey, small brightness range, no texture | dark belt, white, light grey, beige, procedural speckle or noise textures, random colour |

Fruit rotation is random in both settings because a fixed pose would make the narrow set unrealistically easy to tell apart.

Resting pose: see D3.

**Confound note.** Narrow and wide now differ only in how wide each range is, so Step 3.2 measures the effect of spread. But the narrow background was chosen from the real train sample (section 2), so narrow is already matched to how the real photos look. A narrow-vs-wide gap therefore shows what extra spread adds on top of a scene already matched to the real data, not what matching the background is worth. It also means narrow is not a model of a real belt camera.

## 7. Step 2.4: Batch generation

- First `uv run python src/make_scene_params.py --config blender/config.yaml --randomisation wide --n-per-grade 1000 --seed 42` samples every image's parameters and writes `labels.csv` and the splits (D1). No rendering happens here.
- Then `blender -b -P blender/generate.py -- --labels data/synthetic/wide/labels.csv --start 0 --count 500` renders those rows. A small shell loop runs the chunks.
- **Output folder:** `data/synthetic/<randomisation>/<grade>/<grade>_<index>.jpg`. Narrow and wide stay apart so Step 3.2 can train on either.
- **Seeds:** image `i` uses `random.Random(seed * 1_000_000 + i)`. Any single image can be re-rendered exactly, and chunks give the same result whatever order they run in.
- **`labels.csv`** in `data/synthetic/<randomisation>/`, one row per image.
  - Columns: `path, grade, defect_type, seed`, then every sampled parameter (hue, saturation, light energy, camera tilt, background type, and so on).
  - `path` and `grade` use the same names as the real split CSVs, so `OrangeDataset` reads synthetic data unchanged.
- **Splits:** `data/synthetic/<randomisation>/splits/{train,val}.csv`, 85/15, stratified by grade, seed 42. There is no synthetic test split, because the test set is always the real one.
- **Grid:** 16 renders (mixed grades, wide setting) saved to `results/sample_synthetic.png`, with the grade as each title.
- **Real vs synthetic grid:** for each grade, one row of real-train images and one row of synthetic images, saved to `results/real_vs_synthetic_<randomisation>.png`. It shows by eye where the renders differ from the real data.
- **Hue and saturation histograms:** real-train vs synthetic, one panel per grade, saved to `results/hue_sat_<randomisation>.png`. They use about 500 random images per grade and source. Only pixels with saturation above 0.25 are counted, as a rough fruit mask, so white backgrounds do not swamp the histogram. These checks use the real train split only, never val or test.
- The three figures come from one small script, `src/synthetic_report.py`, run after rendering.
- **Duplicate check (D8):** run the Stage 1 phash check once on the first wide batch and report how many groups it finds.

## 8. Decisions

Options are kept for the record. The chosen option is on the **Decided** line.

**D1. Where parameter sampling and labels.csv happen** (affects 2.1)

Blender's bundled Python (3.13) has numpy but not pyyaml.
- **(a) All inside generate.py; install pyyaml into Blender's Python** (`/Applications/Blender.app/.../python -m pip install pyyaml`).
  - Pro: one script, matches the planned structure.
  - Con: it changes the Blender app and is lost on a Blender update. It is also a setup step a reviewer must repeat.
- **(b) Two scripts.** `src/make_scene_params.py` (uv, normal Python) reads config.yaml, samples every image's parameters and writes `labels.csv` and the splits first. generate.py then only reads labels.csv and renders each row.
  - Pro: labels are fixed before any render, which makes "label from parameters" literally true and easy to show. Sampling can be tested without Blender. No change to Blender.
  - Con: one extra file.
- **(c) generate.py reads a JSON config instead of YAML.**
  - Pro: no install, one script.
  - Con: departs from CLAUDE.md's config.yaml, and JSON has no comments to explain the ranges.

**Decided: (b).** `src/make_scene_params.py` is added to the planned structure.

**D2. Images per grade** (affects 2.4)
- **(a) 1,000 per grade** (3,000 per setting, about 50 min each).
  - Pro: fast. Enough for fine-tuning a pretrained ResNet18.
  - Con: the training set is smaller than the real baseline's 6,300, so the comparison is less like-for-like.
- **(b) 3,000 per grade** (9,000 per setting, about 2.5 h each).
  - Pro: matches the real data's size.
  - Con: about 5 h for both settings.

**Decided: (a).** Render more only if val curves suggest it would help.

**D3. Resting pose** (affects 2.3)
- **(a) Random rotation, then move the fruit down so its lowest vertex touches the plane.**
  - Pro: deterministic, adds no time, simple.
  - Con: the pose can be one a real fruit would roll out of, though barely visible on a near-sphere.
- **(b) Rigid-body drop onto the plane** (about 40 simulated frames per image).
  - Pro: natural resting poses and an interview talking point.
  - Con: slower, more code, and little visible gain for a near-spherical fruit.

**Decided: (a).** Say so plainly in the README.

**D4. What "narrow" means** (affects 2.3 and 3.2)
- **(a) Narrow is the plain "factory belt" scene:** dark belt, top-down, fixed light.
  - Pro: tells a clear story: a generator built like the real belt fails on these photos, and randomisation closes the gap.
  - Con: narrow differs from wide in background as well as in spread, so the ablation mixes two effects.
- **(b) Narrow uses a white background,** with everything else tight.
  - Pro: isolates the effect of the width of randomisation.
  - Con: it is tuned toward the test set by looking at it. It is also less like a real belt.

**Decided: (b).** Narrow is a light background with everything else tight; wide is as in section 6. The confound note in section 6 goes into the Stage 3 results.

**D5. Base colour of defective fruit** (affects 2.2)
- **(a) Orange only.**
  - Pro: simple.
  - Con: real cankered and rotten fruit is often greenish, so the model may call it green.
- **(b) Sample the base hue from the fresh and green ranges combined,** then add the defect.
  - Pro: forces the model to use the defects, not colour, to find defective fruit.
  - Con: a mildly defective greenish fruit is close to the green grade, so it is harder to learn.

**Decided: (b).**

**D6. Stem or calyx on every fruit** (affects 2.1)
- **(a) Add a small dark stem disk on all grades.**
  - Pro: in the real train sample (section 2), stems, calyx and navels are visible on fresh and green fruit, not only on defective. If only defect marks were dark, the model could learn "any dark mark means defective". A stem on every grade teaches that a small dark mark at the pole is normal.
  - Con: a bit more geometry and code.
- **(b) No stem.**
  - Pro: simpler.

**Decided: (a).** Justified in decisions.md from the train-sample observation, not from Stage 1 test errors, so no design choice is based on the test set.

**D7. Image format** (affects 2.4)
- **(a) JPEG at quality 95.**
  - Pro: the real images are JPEG. Matching the compression removes one easy cue the model could use to tell sim from real. Files are small.
- **(b) PNG.**
  - Pro: lossless.
  - Con: about 5x the disk space, and a different compression look from the test set.

**Decided: (a).**

**D8. Duplicate check on synthetic images** (affects 2.4)

CLAUDE.md says to check for near-duplicates before training.
- **(a) Skip it, and record why in decisions.md.** Every render is an independent draw with its own seed, so there are no augmented copies to leak.
- **(b) Run the same phash check anyway.**
  - Pro: follows the rule to the letter.
  - Con: about 5 min of hashing that should find nothing. Similar-looking renders are not the same object.

**Decided: (b) once on the first batch, to confirm it finds nothing, then (a).** Cheap evidence beats an argument.

## 9. Honest limits (for the README)

- The peel, defects and lighting are procedural approximations, not measured from real fruit.
- The resting pose is placed, not simulated (D3a). Even a rigid-body drop would not be DEM.
- Neither setting models a real grading machine. Narrow is matched to the look of the real photos (light background), and the dark belt appears only as one option in wide.

## 10. Out of scope for Stage 2

- No training (Stage 3).
- No multiple fruit per image, no motion blur.
- No image-based backgrounds or HDRI downloads.
- No physically measured peel materials (listed as next steps in 3.4).

**Note for Stage 3:** each Stage 3 run must state whether its checkpoint was chosen on synthetic val or real val, and write it in the experiments.csv `notes` column. Choosing on real val uses real labels, so it is not "synthetic only" in the strict sense, and the result must be read that way.
