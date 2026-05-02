# DisasterView — Dataset Statistics Report

*Generated automatically from pipeline outputs. Version 4.0, May 2026.*

---

## 1. Dataset Overview

| Metric                    | Value     |
|--------------------------|-----------|
| Total frames             | 32,233    |
| Annotated frames         | 32,147    |
| Annotation coverage      | 99.7%     |
| CLIP-verified frames     | 29,432    |
| Unique source videos     | 842       |
| Rejected videos          | 846       |
| Disaster types           | 4         |
| Semantic classes         | 10        |
| Total polygon instances  | 77,754    |

---

## 2. Per-Type Breakdown

| Disaster Type | Videos | Frames  | Avg Frames/Video |
|--------------|--------|---------|-----------------|
| earthquake   | 80     | 2,903   | 36.3            |
| flood        | 208    | 7,722   | 37.1            |
| tornado      | 334    | 13,406  | 40.1            |
| wildfire     | 220    | 8,202   | 37.3            |
| **Total**    | **842**  | **32,233** | **38.3** |

---

## 3. Class Distribution

### Instance Counts and Polygon Area

| ID | Class            | Instances | % Instances | % Poly Area |
|----|-----------------|-----------|-------------|-------------|
|  0 | background       |  9,365    |    12.0%    |    14.4%    |
|  1 | building_damaged | 18,497    |    23.8%    |    24.0%    |
|  2 | building_intact  |  3,528    |     4.5%    |     3.1%    |
|  3 | debris_rubble    | 11,677    |    15.0%    |    10.8%    |
|  4 | fire_smoke       |  3,510    |     4.5%    |     6.7%    |
|  5 | road_blocked     |  4,169    |     5.4%    |     4.5%    |
|  6 | road_clear       |  1,782    |     2.3%    |     2.4%    |
|  7 | vegetation       |  3,131    |     4.0%    |     4.0%    |
|  8 | vehicle          |  1,810    |     2.3%    |     2.3%    |
|  9 | water_flood      | 20,285    |    26.1%    |    27.7%    |
| — | **Total**        | **77,754**|             |             |

**Notes:**
- Polygon area is computed via the shoelace formula on normalized [0,1] YOLO-seg coordinates.
- `building_damaged` and `water_flood` together account for ~52% of annotated area.
- `road_clear`, `vehicle`, and `building_intact` are rare and may require class-weighted training.

---

## 4. Train / Val / Test Split Statistics

*Splits are video-disjoint — frames from a given video appear in exactly one split.*

| Split | Videos | Frames  | % of Total |
|-------|--------|---------|-----------|
| train | 588    | 22,869  | 70.9%     |
| val   | 126    | 4,781   | 14.8%     |
| test  | 128    | 4,583   | 14.2%     |

*Source: `split_manifest.json`, seed=42, 70/15/15 ratio applied at video level.*

---

## 5. Annotation Quality (CLIP Verification)

### Overall

| Metric                  | Value    |
|------------------------|----------|
| Mean confidence         | 0.2492   |
| Median confidence       | 0.2487   |
| Std dev                 | 0.0213   |
| Min / Max               | 0.1628 / 0.3122 |
| Frames flagged (< 0.22) | 2,586 (8.7%) |

*CLIP confidence is raw cosine similarity between the frame and its assigned class text prompt.
The range ~0.16–0.31 reflects CLIP's typical output for aerial imagery against short descriptions.*

### Per Disaster Type

| Type         | Frames  |  Mean  | Flagged             |
|-------------|---------|--------|---------------------|
| earthquake   |  2,722 | 0.2428 |   403 (14.8%) |
| flood        |  7,103 | 0.2642 |   248 (3.5%) |
| tornado      | 12,178 | 0.2497 |   679 (5.6%) |
| wildfire     |  7,670 | 0.2366 |  1241 (16.2%) |

---

## 6. Video Collection and Filtering

### Videos Collected vs Rejected

| Disaster Type | Collected | Kept  | Rejected | Rejection Rate |
|--------------|-----------|-------|----------|----------------|
| earthquake   | 384       | 75    | 309      | 80.5%          |
| flood        | 407       | 192   | 215      | 52.8%          |
| tornado      | 432       | 302   | 130      | 30.1%          |
| wildfire     | 397       | 205   | 192      | 48.4%          |
| **Total**    | **1,620** | **774** | **846** | **52.2%** |

### Rejection Reasons

| Reason                | Count | % of Rejected |
|----------------------|-------|---------------|
| Not aerial footage    | 532   | 62.9%         |
| Too blurry            | 314   | 37.1%         |

*Rejection thresholds: CLIP aerial score < 0.25; Laplacian variance < 100 (< 50 for wildfire).*

---

## 7. Geographic Distribution

Exact geographic coordinates were not captured during collection. Videos were sourced via
English-language YouTube search queries. Based on video titles and uploaders in
`video_provenance.csv`, the dataset draws from disasters globally, with likely representation
bias toward North America, Europe, and East Asia due to YouTube's content distribution.

---

## 8. Temporal Distribution

Source videos span approximately 2017–2025 based on YouTube upload dates in `video_provenance.csv`.
The dataset captures a range of recent disaster events rather than a single incident.

---

## 9. Frame Extraction Statistics

| Metric                    | Value    |
|--------------------------|----------|
| Avg frames per video     | 38.3     |
| PySceneDetect threshold  | default (content-aware) |
| DINOv2 dedup threshold   | cosine similarity > 0.97 |
| Frame resolution         | variable (source video resolution) |

*Frames are extracted at scene boundaries and deduplicated using DINOv2 ViT-B/14 embeddings.
Near-duplicate frames (cosine similarity > 0.97) are discarded.*
