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
| defective | per defect type, from the matching real class (D5): blackspot orange, canker yellow-olive, rot orange to yellow | one defect type plus a global shift |

Defect types (one per defective image, chosen evenly):
- **spots:** 3 to 30 small dark brown or black spots from a thresholded Voronoi texture, plus a darker bump in the spot, on a shrivelled peel: a `wrinkle` parameter adds sunken creases (ridged noise, bump only). Matches blackspotted, which is wrinkled in the real images, not smooth.
- **lesions (canker):** many small, raised, irregular light-brown corky marks over the visible face, on a dull yellow-olive base. It uses the same masked Voronoi cells as spots, with the cells distorted by noise, a corky mottled colour, and raised instead of sunken. 11 of 12 real cankered train images look like this; only one showed a few medium patches, and none had the bright yellow halo, so the large-lesion look was dropped.
  - A large-scale noise mask makes the marks cluster unevenly; `canker_density` (per image) sets how much of the face keeps marks. Without it the cells read as a regular polka-dot grid.
  - A yellow-brown scabby mottle (sRGB 0.60, 0.50, 0.25) and rough bump cover the whole fruit, because real canker reads as a rough mottled peel at 224 px, not as separate dots. A grey-tan scab (0.55, 0.50, 0.36) cut canker saturation to 0.70; the yellow-brown one gives 0.74, still below the real 0.78 to 0.87. That is left to be judged on the Step 2.4 histograms.
- **rot:** one or two patches, matched to the real rotten train images.
  - Soft, irregular edge distorted by low-frequency noise.
  - Inside: dark-brown to olive-brown, matte, sunken.
  - A slightly sunken, darker, glossier water-soaked rim just outside the patch.
  - Optional mould that grows out from the patch centre: white at its margin, muted grey-green toward the middle, thinned by fine noise so it looks powdery. With mould_amount 0 there is no mould, which is the most common look in the real images. Known limit: see section 9.

Global shift for every defective image, sampled not fixed:
- Darker: value x (1 - shift_dark).
- Slightly duller: saturation x (1 - shift_dull).
- Higher roughness (less shine).
- No hue change.

Why: real defective train fruit compared with fresh (median of per-image medians) is 0.64x (blackspot), 0.65x (canker) and 0.76x (rot) as bright, 0.89 to 0.97x as saturated, and not browner in hue. Canker is yellower (hue 0.110 vs 0.082); that comes from canker's own base hue range, not from the shift. An earlier version pulled hue toward brown, which the data contradicts.

Caps: shift_dark <= 0.65 and shift_dull <= 0.05, enforced by asserts in generate.py. (The dull cap was 0.12 until the third wide pilot showed synthetic defective/fresh saturation of 0.84 against real 0.89 to 0.97; the dark cap gave a value ratio of 0.67, matching real.) Dulling stays small because desaturating a light greenish base turns it beige, which would make "beige means defective" a shortcut. Darkening keeps a greenish base olive.

Each shift is sampled from a range that starts at 0, so some defective fruit has no global shift and is defective only because of its local defect. That stops the model from relying on overall dullness alone.

Stage 1 Grad-CAM showed the real model uses broad peel cues, so local spots alone are not enough.

**Defect placement.** Every defect has a centre point. Centres are sampled after the fruit's random rotation, on the camera-facing hemisphere: each centre lies within about 60 degrees of the direction from the fruit centre to the camera. The centre is then converted into the fruit's own coordinates, so the material puts the defect where it was sampled. This means no defective image has its defect hidden on the far side, which would be a defective label on a clean-looking image.
- spots: no centre; the spot texture is masked to the camera-facing cap (within about 60 degrees of the view direction), so spots spread across the visible face like real blackspot.
- lesions and rot: each patch has its own centre, up to 3.

All three defect types live in one material, switched by mix factors, so changing type does not recompile shaders. The material has 3 fixed centre slots.
- An unused slot has its centre at the fruit's centre and radius 0.01. Every surface point is then 1.0 from it, and its furthest reach (1.4 x 0.01 plus at most 0.125 edge wobble) stays far below 1.0, so it can never show. An assert at the top of generate.py checks this.
- Used patches have radius at most 0.7 (chord length, about 41 degrees), so a patch stays a local defect. generate.py asserts this, and the 2.4 sampler must draw radii within it.

The defect type is saved in labels.csv. The label stays the 3-class grade.

## 6. Step 2.3: Scenario matrix in config.yaml

Every random value is a `[min, max]` range in `config.yaml`. Two blocks, `narrow` and `wide`, selected by `randomisation: narrow|wide`.

| Parameter | narrow | wide |
|---|---|---|
| Fruit hue, saturation, value | grade ranges, tight | grade ranges, wider value and saturation |
| Peel bump strength, pit scale | fixed | range |
| Squash (z-scale), lumpiness | small range | larger range |
| Defect size, count | middle of range | full range |
| Light intensity | fixed | 0.8x to 1.25x |
| Light colour temperature | 5500 K | 3500 to 7500 K |
| Light position | fixed | random on a hemisphere above, within 70 degrees of the camera direction (never backlit) |
| Fill light | none | weak neutral-white light at the camera, 0.25 of the key's light on the fruit |
| Camera colour processing (after rendering) | none | contrast and saturation boost per image, same ranges for all grades |
| Camera | top-down, fixed distance | tilt 0 to 60 degrees, distance so fruit fills 55 to 100% of the frame, small off-centre shift |
| Fruit rotation | random (free, costs nothing) | random |
| Background | light surface only: plain white to light grey, small brightness range, no texture | dark belt, white, light grey, beige, procedural speckle or noise textures, random colour |

Fruit rotation is random in both settings because a fixed pose would make the narrow set unrealistically easy to tell apart.

Resting pose: see D3.

**Camera colour processing (Step 2.4 pilots):** real photos contain very saturated pixels that the renders cannot reach; pilot 4 saturation coverage was 28% (defective), 53% (fresh) and 75% (green). A per-image contrast then saturation boost, like a phone camera's processing, is applied after rendering in Python. It is sampled the same way for all grades and is wide only, so the narrow calibration stays valid. `cam_saturation` and `cam_contrast` are drawn last in the sampler, so adding them left every render unchanged. A one-sided boost (saturation 1.0 to 1.25, contrast 1.0 to 1.15) raised defective coverage to 94% but shifted fresh and green up as a whole (fresh 78%, green 44%), so the boost is made two-sided.

**Lighting decisions from the Step 2.4 wide pilots:**
- **Key light within 70 degrees of the camera direction.** With light and camera azimuth drawn independently, the key light was often behind the fruit from the camera's view and the visible face was in shadow. Wide renders came out at about 0.7x the real brightness for every grade. The real photos are lit from the camera side. The sampler draws the camera first, then redraws the key light direction until it is within 70 degrees.
- **Weak white fill light at the camera (wide only, fill_ratio 0.25).** It lifts the shadowed side. It is a white light rather than more grey ambient, because grey ambient washed out saturation (Step 2.3). Its power is scaled by the squared distance ratio, so fill_ratio is a ratio of light on the fruit whatever the camera distance. Narrow keeps fill_ratio 0 so its colour calibration stays valid.
- **Light strength 0.8 to 1.25x (320 to 500 W).** Narrowing strength alone did not fix the darkness; it was the geometry. Exposure was not normalised on fruit pixels, because that would erase the real brightness difference between green and fresh fruit.
- Pilot 3 result: green value 0.41 vs real 0.42; fresh 0.69 vs 0.86 and defective 0.46 vs 0.58, both up from pilot 2.

**Decisions from Step 2.3 renders:**
- **Green hue:** wide green material hue is 0.145 to 0.205. At 0.14 the floor rendered at hue 0.133, below the real 5th percentile (0.137).
- **Dark greens are kept as hard examples.** Weak light can render a green fruit darker (about 0.31) than any real green (about 0.38 minimum), so it reads brown-olive. No green-only brightness floor is added: lighting is sampled the same for every grade, and a class-specific floor would create a "dark means not green" shortcut.
- **Light colour temperature stays at 4500 to 8500 K in wide.** A brown-olive wide green was tested at its sampled 5315 K and at 6500 K: the rendered hue was the same (0.133), so warm light was not the cause.

**Notes from Step 2.2 for the ranges:**
- **Mould frequency:** in 12 real rotten train images, only 2 showed mould, and those had heavy mould; the rest had a bare sunken patch. The sampler should give most rot images mould_amount 0, and the rest a high amount, rather than spreading mould_amount evenly across all rot.
- **Salmon tint:** shifted orange fruit in the Step 2.2 sheet rendered salmon (for example rot_a: peel about sRGB 0.82, 0.53, 0.34). That came from the hand-picked base values (hue 0.075, saturation 0.9, value 0.9). Set the orange hue, saturation and value ranges by comparing with the real train hue/saturation histograms (Step 2.4), so shifted fruit reads dull orange, not salmon.

**Confound note.** Narrow and wide now differ only in how wide each range is, so Step 3.2 measures the effect of spread. But the narrow background was chosen from the real train sample (section 2), so narrow is already matched to how the real photos look. A narrow-vs-wide gap therefore shows what extra spread adds on top of a scene already matched to the real data, not what matching the background is worth. It also means narrow is not a model of a real belt camera.

## 7. Step 2.4: Batch generation

- First `uv run python src/make_scene_params.py --config blender/config.yaml --randomisation wide --n-per-grade 1000 --seed 42` samples every image's parameters and writes `labels.csv` (D1). No rendering happens here. It refuses to write into a folder that already holds images, because the resume logic below would otherwise pair new parameters with old renders.
- Then `blender -b -P blender/generate.py -- --labels data/synthetic/wide/labels.csv --start 0 --count 500` renders those rows to their `render_path` (under `renders/`). A small shell loop runs the chunks.
- Then `uv run python src/camera_process.py --labels data/synthetic/wide/labels.csv` applies each row's `cam_contrast` and `cam_saturation` and writes the final image to `path`. Narrow has both at 1.0, so the render is copied byte for byte. Both steps skip finished files, so either can be re-run after an interruption, and the processing is never applied twice. Both write under a temporary `.part.jpg` name and rename when complete, so an interrupted run never leaves a truncated image that would be skipped as done.
- Then `uv run python src/split_synthetic.py --synthetic-dir data/synthetic/wide_v2` writes the splits, once all images exist (below).
- **Output folder:** `data/synthetic/<setting>/<grade>/<grade>_<index>.jpg`, where the setting folder is new for every batch (`narrow_v2`, `wide_v2` after the review fixes). Narrow and wide stay apart so Step 3.2 can train on either.
- **Seeds:** image `i` uses `random.Random(seed * 1_000_000 + i)`. Any single image can be re-rendered exactly, and chunks give the same result whatever order they run in.
- **`labels.csv`** in `data/synthetic/<randomisation>/`, one row per image.
  - Columns: `path, grade, defect_type, seed`, then every sampled parameter (hue, saturation, light energy, camera tilt, background type, and so on).
  - `path` and `grade` use the same names as the real split CSVs, so `OrangeDataset` reads synthetic data unchanged.
- **Splits:** `data/synthetic/<setting>/splits/{train,val}.csv`, 85/15, stratified by grade and defect type, seed 42. Wide keeps phash duplicate groups (D8b, threshold 24) inside one split. Narrow is split without groups, because its fixed framing chains most images into one phash group (section 9); narrow val is used only to watch training curves, never to pick a checkpoint. There is no synthetic test split, because the test set is always the real one.
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

**Decided: (b), then revised in Step 2.3 to follow the real data.** The real train split has no deep-green defective fruit: grade-level defective hue is 0.071 to 0.133 (5th to 95th percentile of per-image medians), cankered up to 0.145. So the defective base hue is not fresh plus green combined; it is set per defect type from the matching real class, which still includes greenish-yellow canker. In config.yaml these become material values (wide 0.07 to 0.152 across the three types), because rendering lowers orange hue by about 0.01.

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
- Mould stays inside the rot patch and ends at a fairly sharp edge. Real mould (2 of 12 sampled rotten train images) spreads over up to half the fruit with a soft white fringe. The synthetic mould is powdery inside but does not reproduce that spread.

- Colour match to the real train split, measured on the full wide batch (3,000 images; 500 per grade on light or dark-belt backgrounds against 500 real train images per grade; coverage = % of real train images whose median lies inside the synthetic 5th to 95th percentile). Pilot 5 numbers in brackets:
  - Hue coverage 91% (87) fresh, 100% (100) green, 94% (90) defective. Saturation 98% (98), 100% (100), 93% (93). Value 96% (93), 99% (95), 84% (77).
  - Defective brightness is the main gap: synthetic median 0.43 against real 0.58, coverage 84%; the brightest real defective fruit is not reached. Defective saturation is also lower (median 0.72 against 0.82): the histogram lacks the real peak at 0.85 to 0.95.
  - Fresh is darker overall: median value 0.71 against real 0.86, although 96% of real fresh images are still covered.
  - Green hue median is 0.169 against real 0.180; coverage is 100%.
  - Narrow is deliberately tight, so its coverage of the real 5th to 95th spread is low by design (hue 65/60/70%, saturation 29/29/63%, value 45/56/90% for fresh/green/defective). Its medians: fresh value 0.82 (real 0.86), green hue 0.169 (0.180), defective value 0.46 (0.58).
  - Narrow calibration (measured on 30 test renders): green, blackspot and canker saturation sit 0.01 to 0.04 outside the real 25th to 75th percentile; canker is the furthest (0.74 against 0.78 to 0.87).
  - These are colour statistics only. They say nothing about shape, texture or defect appearance, which were checked by eye.
- Near-duplicate check (D8b) on the full batch: wide has 26 phash groups of 2 or 3 images (54 images), and 14 of them mix grades, so they are similar poses and framings, not repeated renders. In narrow, phash links 2,376 of 3,000 images into chains (largest 2,328, all three grades): every narrow image has the same centred framing on the same light surface, and phash compares grey-level structure, not colour. Pairs inside the big group still differ by 7 to 38 grey levels per pixel on average, and no two files in either set are identical. So phash is not a useful duplicate test for narrow, and narrow train/val images are far more alike than wide ones; that is the point of the narrow setting, but it makes narrow synthetic val accuracy optimistic.
- Framing: `camera_height` places the camera as if the fruit were a flat disc of diameter 2R. A sphere seen in perspective looks wider than that, so at wide fill 1.0 the fruit spans about 107% of the frame width and is cut off at the sides, more so with lens shift. About a quarter of wide renders have fill above 0.9 and touch or cross the frame edge. Real photos often fill the frame too, so this was left as is. Narrow (fill 0.85) spans about 90%.
- Camera contrast (wide only, 0.95 to 1.15) pivots around the mean grey level of the whole image, background included. Fruit on a dark belt is pushed slightly brighter and fruit on a white surface slightly darker, so the boost adds a small link between fruit brightness and background type.

## 10. Out of scope for Stage 2

- No training (Stage 3).
- No multiple fruit per image, no motion blur.
- No image-based backgrounds or HDRI downloads.
- No physically measured peel materials (listed as next steps in 3.4).

**Note for Stage 3:** each Stage 3 run must state whether its checkpoint was chosen on synthetic val or real val, and write it in the experiments.csv `notes` column. Choosing on real val uses real labels, so it is not "synthetic only" in the strict sense, and the result must be read that way.

**Note for Stage 3 (from the Step 2.4 pilot):** real photos contain very saturated pixels (0.9 to 1.0) that the renders do not produce, probably from the phone camera's colour processing. Use mild colour jitter on synthetic training images (saturation and contrast, no hue) to cover that range.
