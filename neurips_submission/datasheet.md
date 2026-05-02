# Datasheet for DisasterView

*Following Gebru et al. (2018), "Datasheets for Datasets."*

---

## Motivation

**For what purpose was the dataset created?**
DisasterView was created to advance semantic segmentation of UAV/drone aerial footage captured during natural disasters. Existing aerial datasets focus on detection or classification rather than dense pixel-level segmentation, and none provide multi-disaster-type coverage at scale. The dataset supports development of automated disaster-assessment models that could aid first responders.

**Who created the dataset and on behalf of which entity?**
Created by researchers at Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), Abu Dhabi. This work is being submitted to the NeurIPS 2026 Datasets and Benchmarks Track.

**Who funded the creation of the dataset?**
Internal research funding from MBZUAI.

**Any other comments?**
The dataset is released under CC BY 4.0. Users should be aware that source videos remain subject to YouTube's Terms of Service and individual creator copyrights. We distribute extracted frames under a fair-use research exemption; users intending commercial use should re-verify licensing for each source video using the provided `video_provenance.csv`.

---

## Composition

**What do the instances that comprise the dataset represent?**
Each instance is a single JPEG video frame extracted from a YouTube UAV/drone video of a natural disaster, accompanied by a YOLO-segmentation annotation file (.txt) containing polygon masks for one or more of 10 semantic classes.

**How many instances are there in total?**
29,673 annotated frames from 774 unique videos across 4 disaster types:

| Disaster Type | Videos | Frames |
|--------------|--------|--------|
| earthquake   | 75     | 2,722  |
| flood        | 192    | 7,103  |
| tornado      | 302    | 12,178 |
| wildfire     | 205    | 7,670  |

**Does the dataset contain all possible instances or is it a sample?**
It is a curated sample of publicly available YouTube videos. The collection is bounded by the reach of four YouTube search queries (one per disaster type) using yt-dlp, filtered by aerial quality and visual clarity.

**What data does each instance consist of?**
- A JPEG frame (variable resolution, sourced from UAV video)
- A YOLO-seg annotation file with normalized polygon coordinates and class IDs
- A row in `quality_report.csv` with CLIP verification confidence score
- A row in `split_manifest.json` assigning the source video to train/val/test

**Is there a label or target associated with each instance?**
Yes. Each frame has pixel-level segmentation annotations for 10 classes:

| ID | Class            |
|----|-----------------|
|  0 | background       |
|  1 | building_damaged |
|  2 | building_intact  |
|  3 | debris_rubble    |
|  4 | fire_smoke       |
|  5 | road_blocked     |
|  6 | road_clear       |
|  7 | vegetation       |
|  8 | vehicle          |
|  9 | water_flood      |

**Is any information missing from individual instances?**
Geographic coordinates and precise event timestamps are not available. Video resolution varies across source videos (typically 720p–4K).

**Are there recommended data splits?**
Yes. `split_manifest.json` provides **video-disjoint** train/val/test splits (70/15/15 by video count). This is critical: Roboflow's internal split is random and causes data leakage. Always use the manifest.

| Split | Videos | Frames |
|-------|--------|--------|
| train | 540    | 20,791 |
| val   | 116    | 4,429  |
| test  | 118    | 4,453  |

**Are there any errors, sources of noise, or redundancies in the dataset?**
Annotations are generated automatically via CLIP scoring and k-means segmentation — they are not manually verified. CLIP verification flagged 2,586 of 29,673 frames (8.7%) as low-confidence (score < 0.22); these remain in the dataset but are identified in `quality_report.csv`.

**Is the dataset self-contained, or does it link to or otherwise rely on external resources?**
The dataset is self-contained as distributed. Source video URLs are recorded in `video_provenance.csv`; the original YouTube videos may be removed by uploaders at any time.

**Does the dataset contain data that might be considered confidential?**
No. All source videos were publicly posted on YouTube.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**
The dataset contains imagery of destruction, damaged buildings, flooding, and wildfire. Some frames may depict scenes that viewers find distressing.

---

## Collection Process

**How was the data associated with each instance acquired?**
Videos were collected via automated YouTube search using yt-dlp with four queries:
- `"drone footage earthquake damage 2024"` (earthquake)
- `"UAV aerial flood disaster footage"` (flood)
- `"drone tornado damage footage"` (tornado)
- `"aerial wildfire drone footage"` (wildfire)

Each video was then quality-filtered before frame extraction.

**What mechanisms or procedures were used to collect the data?**
1. **Download**: yt-dlp downloads up to 600 videos per disaster type
2. **Quality filter**: CLIP aerial-scene similarity (threshold 0.25) and Laplacian sharpness (threshold 100; 50 for wildfire due to smoke haze)
3. **Frame extraction**: PySceneDetect scene-boundary detection; DINOv2 cosine-similarity deduplication (threshold 0.97) removes near-duplicate frames
4. **Annotation**: CLIP text-image scoring over 10 class prompts; k-means segmentation generates polygon masks (~0.13 s/frame)
5. **Verification**: CLIP re-scores each annotated frame; frames below confidence 0.22 are flagged in `quality_report.csv`

**Who was involved in the data collection process and how were they compensated?**
Fully automated pipeline — no human annotators. Researchers at MBZUAI designed and ran the pipeline.

**Over what timeframe was the data collected?**
Pipeline was executed in April 2026. Source videos span approximately 2017–2025 based on upload dates.

**Were any ethical review processes conducted?**
The collection uses only publicly available YouTube videos. No personally identifiable information is collected. No IRB review was required.

**Does the dataset relate to people?**
Incidentally. Some frames may contain vehicles or human figures at UAV altitude. No face recognition, identification, or biometric data is collected or annotated.

**Did you collect the data from the individuals in question directly, or obtain it via third parties?**
Via YouTube, a third-party platform. All collected videos were publicly accessible at time of download.

**Were the individuals in question notified about the data collection?**
Not applicable — source videos are publicly posted content on YouTube.

---

## Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes — full pipeline details are in the Collection Process section. In summary:
- Blurry and non-aerial videos were rejected (846 total: 314 too blurry, 532 not aerial)
- Near-duplicate frames were removed via DINOv2 similarity
- Annotations are machine-generated; no manual correction was performed

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?**
Source videos are stored locally. The distributed dataset contains only the extracted frames and annotations.

**Is the software that was used to preprocess/clean/label the instances available?**
Yes. The full pipeline is available at the project repository as `pipeline.py`.

**Any other comments?**
The automatic annotation approach trades precision for scale. CLIP-based class assignment is competitive but may assign incorrect dominant classes in ambiguous scenes. Users requiring high-precision annotations should consider using `quality_report.csv` to filter frames and manually correcting low-confidence annotations.

---

## Uses

**Has the dataset been used for any tasks already?**
The dataset was used internally at MBZUAI to train a SegFormer-B0 baseline for the NeurIPS 2026 submission.

**Is there a repository that links to any or all papers or systems that use the dataset?**
To be maintained at the project's HuggingFace page upon public release.

**What (other) tasks could the dataset be used for?**
- Semantic segmentation of disaster imagery
- Multi-class change detection (pre/post disaster, with external pre-disaster data)
- Damage severity classification
- Scene understanding for autonomous disaster-response UAVs
- Benchmarking aerial image segmentation models

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
- Class imbalance: `water_flood` (27.7% pixel area) and `building_damaged` (24.0%) dominate; `road_clear`, `vehicle`, and `building_intact` are rare
- Geographic bias: videos collected via English-language YouTube queries; may underrepresent disasters in non-English-speaking regions
- Disaster type imbalance: tornado (12,178 frames) vs. earthquake (2,722 frames)
- Annotation noise from automated pipeline

**Are there tasks for which the dataset should not be used?**
- Identifying individuals from aerial footage
- Any commercial application without verifying source video licensing
- As ground truth for operational (non-research) disaster systems without expert validation

---

## Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?**
Yes. Public release planned on HuggingFace upon NeurIPS acceptance.

**How will the dataset be distributed?**
HuggingFace Datasets hub with Croissant metadata. A download script will also be provided.

**When will the dataset be distributed?**
Upon NeurIPS 2026 camera-ready acceptance (expected late 2026).

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**
The annotations, metadata, and extracted frames are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Source videos remain subject to the original creators' licenses and YouTube ToS.

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**
Individual YouTube video creators may have reserved rights. We release extracted frames under research fair use.

**Do any export controls or other regulatory restrictions apply to the dataset?**
No.

---

## Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
MBZUAI research team. Contact: the corresponding author (see paper).

**How can the owner/curator/manager of the dataset be contacted?**
Via the HuggingFace dataset repository issue tracker, or by email (listed in the paper).

**Is there an erratum?**
Not at time of release. Errata will be posted to the HuggingFace repository.

**Will the dataset be updated?**
Yes. The pipeline is designed to support incremental expansion toward 1 million frames. Dataset versions are tracked via Roboflow.

**Will older versions of the dataset continue to be supported/hosted/maintained?**
Prior versions will remain accessible on Roboflow and HuggingFace with version tags.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Yes. The full pipeline (`pipeline.py`) is open-sourced. Contribution guidelines will be published in the repository.

**Any other comments?**
We recommend using `split_manifest.json` for all train/val/test partitioning rather than Roboflow's built-in split, which is not video-disjoint and causes data leakage.
