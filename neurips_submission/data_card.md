# DisasterView — Data Card

*Google Model Cards format adapted for datasets.*

---

## Dataset Summary

DisasterView is a large-scale UAV/drone aerial segmentation dataset for natural disaster scenes.
It contains **29,673 annotated frames** from **774 unique videos** spanning
four disaster categories with pixel-level polygon masks for 10 semantic classes.

| Attribute        | Value |
|-----------------|-------|
| Frames           | 29,673 |
| Videos           | 774 |
| Disaster types   | earthquake, flood, tornado, wildfire |
| Semantic classes | 10 |
| Annotation type  | Polygon segmentation (YOLO-seg format) |
| Split strategy   | Video-disjoint (70 / 15 / 15) |
| License          | CC BY 4.0 |
| Version          | 3.0 (April 2026) |

---

## Intended Use

**Primary intended use:** Training and evaluating semantic segmentation models on aerial disaster
imagery. Designed for researchers developing automated damage-assessment systems for UAV platforms.

**Primary intended users:** Computer vision researchers, disaster-response AI developers, NLP/CV
benchmark practitioners.

**Out-of-scope uses:**
- Operational (non-research) disaster-response systems without expert validation
- Identifying individuals from aerial imagery
- Any use requiring individual video creator permission beyond research fair use

---

## Dataset Factors

### Disaster Types
Four natural disaster categories with varying visual characteristics:

| Type       | Videos | Frames  | Distinguishing features |
|-----------|--------|---------|------------------------|
| earthquake | 75     | 2,722   | Structural collapse, rubble, dust clouds |
| flood      | 192    | 7,103   | Standing water, submerged structures |
| tornado    | 302    | 12,178  | Debris fields, uprooted vegetation, structural damage |
| wildfire   | 205    | 7,670   | Active fire, smoke plumes, burn scars |

### Geographic Coverage
Globally sourced via YouTube. English-language queries may bias toward North American, European,
and East Asian disasters. Exact geographic coordinates are not available; see `video_provenance.csv`
for video titles and uploaders which may indicate location.

### Temporal Coverage
Source videos uploaded approximately 2017–2025. See `video_provenance.csv` for per-video upload dates.

### Video Quality
- Resolution: 720p–4K (variable)
- Viewpoint: Aerial/nadir-to-oblique (all frames verified via CLIP aerial-scene scoring)
- Blur rejection: Laplacian variance threshold 100 (50 for wildfire)

---

## Metrics

Recommended evaluation metric: **mean Intersection over Union (mIoU)** across all 10 classes,
computed using the video-disjoint test split from `split_manifest.json`.

Secondary metrics:
- Per-class IoU (especially for rare classes: road_clear, vehicle, building_intact)
- Frequency-weighted IoU (accounts for class imbalance)
- Boundary IoU (structural accuracy of polygon predictions)

**Important:** Do NOT evaluate using Roboflow's built-in train/valid/test folders — they are
randomly split and cause data leakage. Use `split_manifest.json` exclusively.

---

## Evaluation Data

Test split: **118 videos, ~4,453 frames** (video-disjoint — no frames from test videos appear
in training data).

Per-type test representation mirrors the overall dataset distribution.

---

## Training Data

Training split: **540 videos, ~20,791 frames**.
Validation split: **116 videos, ~4,429 frames**.

Class distribution in training data mirrors the full dataset (see Quantitative Analyses below).
No data augmentation was applied; frames represent real UAV footage conditions.

---

## Quantitative Analyses

### Class Distribution

| ID | Class            | Instances | % Instances | % Area  |
|----|-----------------|-----------|-------------|---------|
|  0 | background       | 9,365     | 12.0%       | 14.4%   |
|  1 | building_damaged | 18,497    | 23.8%       | 24.0%   |
|  2 | building_intact  | 3,528     |  4.5%       |  3.1%   |
|  3 | debris_rubble    | 11,677    | 15.0%       | 10.8%   |
|  4 | fire_smoke       | 3,510     |  4.5%       |  6.7%   |
|  5 | road_blocked     | 4,169     |  5.4%       |  4.5%   |
|  6 | road_clear       | 1,782     |  2.3%       |  2.4%   |
|  7 | vegetation       | 3,131     |  4.0%       |  4.0%   |
|  8 | vehicle          | 1,810     |  2.3%       |  2.3%   |
|  9 | water_flood      | 20,285    | 26.1%       | 27.7%   |
| — | **Total**        | **77,754**|             |         |

### Annotation Quality

| Metric                  | Value  |
|------------------------|--------|
| Mean CLIP confidence    | 0.2492 |
| Median CLIP confidence  | 0.2487 |
| Std dev                 | 0.0213 |
| Min / Max               | 0.1628 / 0.3122 |
| Flagged (< 0.22)        | 2,586 (8.7%) |

### Dataset Splits

| Split | Videos | Frames  | % Frames |
|-------|--------|---------|----------|
| train | 540    | 20,791  | 70.1%    |
| val   | 116    | 4,429   | 14.9%    |
| test  | 118    | 4,453   | 15.0%    |

---

## Ethical Considerations

**Privacy:** Source videos are publicly posted on YouTube. Frames may incidentally contain
vehicles or persons at UAV altitude. No biometric data is extracted or annotated.

**Dual use risk:** Automated damage-assessment systems trained on this data could be misused
for surveillance. We recommend disclosure in downstream work.

**Copyright:** Extracted frames are distributed under research fair use (CC BY 4.0 for
annotations). Commercial users must verify licensing for each source video via `video_provenance.csv`.

**Representation bias:** English-language YouTube queries may underrepresent disasters in the
Global South. Communities affected by disasters in non-English-speaking regions may be
systematically underrepresented.

**Human subjects:** Not applicable. No personally identifiable information is annotated.

---

## Caveats and Recommendations

1. **Use split_manifest.json** — never Roboflow's random split.
2. **Class imbalance** — consider class-weighted loss functions; `road_clear` and `vehicle` have
   very few instances.
3. **Annotation noise** — automated pipeline; consider filtering by CLIP confidence for
   high-precision applications.
4. **Tornado over-representation** — 41% of frames are tornado. Cross-type evaluations should
   be reported per-type.
5. **Resolution variation** — frames range from 360p to 4K; normalize or resize consistently.
6. **No augmentations** — the dataset intentionally omits augmentations to serve as a clean
   benchmark. Apply augmentations in your training code, not as dataset preprocessing.
