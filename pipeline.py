"""
DisasterView Dataset Pipeline
==============================
Stages:
  1 - YouTube metadata collection (yt-dlp --skip-download; no video files kept)
  2 - Metadata quality pre-filter (duration / resolution from info.json)
  3 - Smart frame extraction (temp download → CLIP+Laplacian check → extract → delete)
  4 - Auto-labeling (CLIP+k-means segmentation + Roboflow upload)
  5 - CLIP quality verification
  6 - Roboflow dataset version generation
  7 - HuggingFace dataset upload (exports + metadata → mahergzani/disasterview)

Disk strategy: video files are NEVER stored permanently.
  - Stage 1 saves .info.json + thumbnail to neurips_submission/video_metadata/<type>/
  - Stage 3 downloads each video to _temp_download.mp4, extracts frames, then deletes it
  - video_provenance.csv is the authoritative source record (updated after every Stage 1 run)

Usage:
  python pipeline.py --stage 1
  python pipeline.py --all
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Load .env before anything else so ROBOFLOW_API_KEY is available everywhere.
# Uses python-dotenv if installed, otherwise falls back to a stdlib-only parser
# so the key is always available regardless of which Python runs this file.
def _load_env_file() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    # stdlib fallback: handles KEY=value, strips quotes, ignores comments
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        os.environ.setdefault(key, val)

_load_env_file()

# ── directory layout ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
VIDEOS_DIR      = ROOT / "videos"          # legacy – not written by new pipeline runs
FRAMES_DIR      = ROOT / "frames"
ANNOTATIONS_DIR = ROOT / "annotations"
REVIEW_DIR      = ROOT / "review"
VERIFIED_DIR    = ROOT / "verified"
NEURIPS_DIR     = ROOT / "neurips_submission"
METADATA_DIR    = NEURIPS_DIR / "video_metadata"   # .info.json + thumbnails per video
PROVENANCE_CSV  = ROOT / "video_provenance.csv"
TEMP_VIDEO      = ROOT / "_temp_download.mp4"      # transient; deleted after each video
LOGS_DIR        = ROOT / "logs"

STATUS_FILE = ROOT / "pipeline_status.json"
LOG_FILE    = ROOT / "pipeline.log"

DISASTER_TYPES = ["earthquake", "flood", "tornado", "wildfire"]

SEARCH_QUERIES = {
    "earthquake": "drone footage earthquake damage 2024",
    "flood":      "UAV aerial flood disaster footage",
    "tornado":    "drone tornado damage footage",
    "wildfire":   "aerial wildfire drone footage",
}
# Extra generic query used for the 5th search slot
EXTRA_QUERY = "DJI drone disaster footage"

TARGET_VIDEOS_PER_TYPE = 800
TARGET_FRAMES_PER_TYPE = 25_000

CLIP_REJECT_THRESHOLD  = 0.25   # score below this → reject
LAPLACIAN_THRESHOLD    = 100.0  # variance below this → too blurry
DINO_SIM_THRESHOLD     = 0.97   # cosine similarity above this → duplicate
CLIP_VERIFY_THRESHOLD  = 0.22   # confidence below this → flag for review (raw cosine sim range ~0.18–0.31)

# Per-type Laplacian overrides (smoke-heavy wildfire aerial is legitimately hazy)
LAPLACIAN_THRESHOLDS: dict = {
    "wildfire": 50.0,
}

# Boost queries used when re-running Stage 1 for earthquake only
EARTHQUAKE_BOOST_QUERIES = [
    "aerial drone earthquake damage site 2024",
    "UAV footage earthquake collapsed buildings",
    "drone survey earthquake destruction",
    "aerial view earthquake rubble 2023",
    "DJI drone earthquake aftermath",
]
EARTHQUAKE_BOOST_PER_QUERY = 15  # downloads per query → up to 75 attempts ≈ 50 new files

LABEL_PROMPTS = [
    "background sky ground",
    "damaged building collapsed structure",
    "intact building standing structure",
    "debris rubble wreckage",
    "fire smoke flames",
    "blocked road obstructed path",
    "clear road open path",
    "vegetation trees plants",
    "vehicle car truck",
    "flood water inundation",
]

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")


# ── status helpers ────────────────────────────────────────────────────────────
def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {str(i): {"complete": False, "timestamp": None} for i in range(1, 8)}


def save_status(status: dict) -> None:
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def mark_complete(stage: int) -> None:
    status = load_status()
    status[str(stage)] = {"complete": True, "timestamp": datetime.now(timezone.utc).isoformat()}
    save_status(status)
    log.info(f"Stage {stage} marked complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 – YouTube Collection
# ─────────────────────────────────────────────────────────────────────────────
def stage1_youtube_collection(type_filter: str = None):
    log.info("=" * 60)
    log.info("STAGE 1 – YouTube Metadata Collection (no video download)")
    log.info("=" * 60)

    try:
        from tqdm import tqdm
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
        from tqdm import tqdm

    if shutil.which("yt-dlp") is None:
        log.info("Installing yt-dlp …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"])

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    types_to_run = [type_filter] if type_filter else DISASTER_TYPES
    total_collected = 0

    for dtype in tqdm(types_to_run, desc="Disaster types", unit="type"):
        out_dir = METADATA_DIR / dtype
        out_dir.mkdir(parents=True, exist_ok=True)

        # Earthquake boost: multiple targeted queries, skip already-seen IDs via archive
        if type_filter == "earthquake" and dtype == "earthquake":
            _run_earthquake_boost(out_dir)
            count = _count_videos(out_dir)
            log.info(f"[earthquake] {count} metadata files after boost.")
            total_collected += count
            continue

        # Standard path: single query per type
        existing = _count_videos(out_dir)
        if existing >= TARGET_VIDEOS_PER_TYPE:
            log.info(f"[{dtype}] Already has {existing} metadata files – skipping.")
            total_collected += existing
            continue

        need  = TARGET_VIDEOS_PER_TYPE - existing
        query = SEARCH_QUERIES.get(dtype, EXTRA_QUERY)
        log.info(f"[{dtype}] Fetching metadata for up to {need} more videos: '{query}'")
        _yt_fetch_metadata(query, out_dir, need)

        count = _count_videos(out_dir)
        log.info(f"[{dtype}] {count} total metadata files.")
        total_collected += count

    _update_provenance_csv()
    log.info(f"Stage 1 complete. Total video metadata records: {total_collected}")
    mark_complete(1)


def _count_videos(directory: Path) -> int:
    """Count .info.json metadata files (one per video) in a metadata directory."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.name.endswith(".info.json"))


def _yt_fetch_metadata(query: str, out_dir: Path, max_results: int,
                       archive: Path = None, extra_args: list = None):
    """Fetch .info.json + thumbnail for up to max_results YouTube videos. No video download."""
    yt_args = [
        "yt-dlp",
        f"ytsearch{max_results * 3}:{query}",
        "--skip-download",
        "--write-info-json",
        "--write-thumbnail",
        "--match-filter", "duration >= 30 & duration <= 600",
        "--output", str(out_dir / "%(id)s.%(ext)s"),
        "--max-downloads", str(max_results),
        "--no-playlist",
        "--ignore-errors",
        "--quiet",
        "--no-warnings",
    ]
    if archive:
        yt_args += ["--download-archive", str(archive)]
    if extra_args:
        yt_args += extra_args
    try:
        subprocess.run(yt_args, check=False, timeout=3600)
    except subprocess.TimeoutExpired:
        log.warning(f"yt-dlp metadata fetch timed out for query: {query}")


def _run_earthquake_boost(out_dir: Path):
    """Fetch metadata for additional earthquake videos using the boost query set."""
    archive = out_dir / ".yt_dlp_archive.txt"
    seen_ids = {jf.stem for jf in out_dir.glob("*.info.json")}
    archive.write_text("\n".join(f"youtube {vid}" for vid in seen_ids) + "\n")
    log.info(f"[earthquake] Boost mode: {len(seen_ids)} existing IDs in archive, "
             f"running {len(EARTHQUAKE_BOOST_QUERIES)} queries × {EARTHQUAKE_BOOST_PER_QUERY} fetches.")
    for query in EARTHQUAKE_BOOST_QUERIES:
        log.info(f"[earthquake] Boost query: '{query}'")
        _yt_fetch_metadata(query, out_dir, EARTHQUAKE_BOOST_PER_QUERY, archive=archive)


# ── provenance helpers ────────────────────────────────────────────────────────

def _parse_info_json(json_path: Path, dtype: str) -> dict | None:
    """Extract provenance fields from a yt-dlp .info.json file."""
    try:
        info = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    vid_id = json_path.stem
    raw_date = info.get("upload_date", "")
    upload_date = (f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                   if raw_date and len(raw_date) == 8 else "")
    w, h = info.get("width", ""), info.get("height", "")
    tags = info.get("tags") or []
    desc = (info.get("description") or "").replace("\n", " ")[:500]
    return {
        "video_id":           vid_id,
        "title":              (info.get("title") or "")[:200],
        "uploader":           info.get("uploader") or "",
        "upload_date":        upload_date,
        "url":                f"https://www.youtube.com/watch?v={vid_id}",
        "duration_seconds":   info.get("duration") or "",
        "resolution":         f"{w}x{h}" if w and h else "",
        "disaster_type":      dtype,
        "view_count":         info.get("view_count") or "",
        "description":        desc,
        "tags":               "|".join(str(t) for t in tags[:10]),
        "geographic_location": info.get("location") or "",
    }


_PROVENANCE_FIELDS = [
    "video_id", "title", "uploader", "upload_date", "url",
    "duration_seconds", "resolution", "disaster_type",
    "view_count", "description", "tags", "geographic_location",
    "kept", "rejection_reason", "frame_count",
]


def _get_rejected_ids() -> set:
    """Return set of video IDs recorded in rejected_videos.csv (handles old and new format)."""
    rejected: set = set()
    reject_csv = ROOT / "rejected_videos.csv"
    if not reject_csv.exists():
        return rejected
    with open(reject_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("video_id"):
                rejected.add(row["video_id"])
            elif row.get("video"):
                stem = Path(row["video"]).stem
                rejected.add(stem.replace(".mp4", "").replace(".webm", "").replace(".mkv", ""))
    return rejected


def _update_provenance_csv():
    """Rebuild video_provenance.csv from all .info.json files + rejected_videos.csv."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    rejected_map: dict = {}
    reject_csv = ROOT / "rejected_videos.csv"
    if reject_csv.exists():
        with open(reject_csv, newline="") as f:
            for row in csv.DictReader(f):
                vid_id = row.get("video_id") or ""
                if not vid_id and row.get("video"):
                    stem = Path(row["video"]).stem
                    vid_id = stem.replace(".mp4", "").replace(".webm", "").replace(".mkv", "")
                if vid_id:
                    rejected_map[vid_id] = row.get("reason", "")

    frame_counts: dict = {}
    if (ROOT / "split_manifest.json").exists():
        try:
            manifest = json.loads((ROOT / "split_manifest.json").read_text())
            for v in manifest.get("videos", {}).values():
                frame_counts[v["video_id"]] = v.get("frame_count", 0)
        except Exception:
            pass

    rows = []
    for dtype in DISASTER_TYPES:
        meta_dir = METADATA_DIR / dtype
        if not meta_dir.exists():
            continue
        for jf in sorted(meta_dir.glob("*.info.json")):
            row = _parse_info_json(jf, dtype)
            if row is None:
                continue
            vid_id = row["video_id"]
            row["kept"]             = "no" if vid_id in rejected_map else "yes"
            row["rejection_reason"] = rejected_map.get(vid_id, "")
            row["frame_count"]      = frame_counts.get(vid_id, 0)
            rows.append(row)

    for dest in [PROVENANCE_CSV, NEURIPS_DIR / "video_provenance.csv"]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_PROVENANCE_FIELDS)
            w.writeheader()
            w.writerows(rows)
    log.info(f"video_provenance.csv updated: {len(rows)} videos → {PROVENANCE_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 – Video Quality Filtering
# ─────────────────────────────────────────────────────────────────────────────
def stage2_quality_filtering(type_filter: str = None):
    """Lightweight metadata pre-filter: rejects videos based on duration and resolution
    extracted from .info.json files. No video download required.
    CLIP + Laplacian quality checks now run inline in Stage 3."""
    log.info("=" * 60)
    log.info("STAGE 2 – Metadata Quality Pre-filter (no video download)")
    log.info("=" * 60)

    types_to_run = [type_filter] if type_filter else DISASTER_TYPES

    reject_csv = ROOT / "rejected_videos.csv"
    preserved_rows: list = []
    if type_filter and reject_csv.exists():
        with open(reject_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("type") != type_filter:
                    preserved_rows.append(row)
        log.info(f"Preserved {len(preserved_rows)} existing rows for non-{type_filter} types.")

    new_rejected_rows: list = []
    kept_total = 0

    for dtype in types_to_run:
        meta_dir = METADATA_DIR / dtype
        if not meta_dir.exists():
            log.warning(f"[{dtype}] No metadata directory – skipping (run Stage 1 first).")
            continue

        json_files = sorted(meta_dir.glob("*.info.json"))
        log.info(f"[{dtype}] Pre-filtering {len(json_files)} metadata records.")

        for jf in json_files:
            try:
                info = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                new_rejected_rows.append({"video_id": jf.stem, "reason": "unreadable info.json", "type": dtype})
                continue

            duration = float(info.get("duration") or 0)
            height   = int(info.get("height") or 0)

            reason = None
            if duration > 0 and (duration < 30 or duration > 600):
                reason = f"duration out of range ({duration:.0f}s, expected 30–600s)"
            elif height > 0 and height < 240:
                reason = f"resolution too low ({height}p < 240p)"

            if reason:
                new_rejected_rows.append({"video_id": jf.stem, "reason": reason, "type": dtype})
                log.info(f"  REJECT {jf.stem}: {reason}")
            else:
                kept_total += 1

    all_rows = preserved_rows + new_rejected_rows
    with open(reject_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "reason", "type"])
        writer.writeheader()
        writer.writerows(all_rows)

    log.info(f"Stage 2 complete. Kept: {kept_total} | New pre-rejections: {len(new_rejected_rows)} "
             f"| Total in CSV: {len(all_rows)}")
    mark_complete(2)


def _check_video(vpath: Path, model, preprocess, text_features, n_aerial: int,
                 device: str, laplacian_threshold: float = None):
    """Return rejection reason string, or None if video passes."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    if laplacian_threshold is None:
        laplacian_threshold = LAPLACIAN_THRESHOLD

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return "cannot open video"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return "zero frames"

    sample_indices = [int(total_frames * i / 5) for i in range(5)]
    laplacians = []
    clip_scores_aerial = []
    clip_scores_reject = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacians.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_features.T).squeeze(0).cpu().numpy()

        clip_scores_aerial.append(float(sims[:n_aerial].max()))
        clip_scores_reject.append(float(sims[n_aerial:].max()))

    cap.release()

    if not laplacians:
        return "could not decode frames"

    avg_lap    = float(np.mean(laplacians))
    avg_aerial = float(np.mean(clip_scores_aerial)) if clip_scores_aerial else 0.0
    avg_reject = float(np.mean(clip_scores_reject)) if clip_scores_reject else 0.0

    if avg_lap < laplacian_threshold:
        return f"too blurry (Laplacian={avg_lap:.1f} < {laplacian_threshold})"
    if avg_aerial < CLIP_REJECT_THRESHOLD:
        return f"not aerial footage (CLIP aerial={avg_aerial:.3f} < {CLIP_REJECT_THRESHOLD})"
    if avg_reject > avg_aerial:
        return f"classified as non-UAV content (reject={avg_reject:.3f} > aerial={avg_aerial:.3f})"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 – Frame Extraction (temp download → quality check → extract → delete)
# ─────────────────────────────────────────────────────────────────────────────
def stage3_frame_extraction(force: bool = False):
    """For each video in METADATA_DIR: download to TEMP_VIDEO, run CLIP+Laplacian
    quality check, extract frames if it passes, then delete TEMP_VIDEO immediately.
    Videos never accumulate on disk."""
    log.info("=" * 60)
    log.info("STAGE 3 – Frame Extraction (temp download, immediate delete)")
    log.info("=" * 60)

    _ensure_packages(["tqdm", "scenedetect[opencv]", "torch", "torchvision", "Pillow",
                      "opencv-python", "git+https://github.com/openai/CLIP.git"])

    import cv2
    import numpy as np
    import torch
    import clip
    from PIL import Image
    from tqdm import tqdm
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # CLIP model for inline quality check (replaces standalone Stage 2 CLIP check)
    log.info("Loading CLIP model for quality check …")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    aerial_prompts = ["aerial drone footage from above", "bird's eye view UAV footage"]
    reject_prompts = ["news broadcast studio", "ground level shot", "slideshow images", "talking head interview"]
    all_text = clip.tokenize(aerial_prompts + reject_prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(all_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    n_aerial = len(aerial_prompts)

    # DINOv2 for frame deduplication
    log.info("Loading DINOv2 model …")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    dino.eval().to(device)
    dino_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Clean up any leftover temp file from a prior interrupted run
    if TEMP_VIDEO.exists():
        TEMP_VIDEO.unlink()

    rejected_ids = _get_rejected_ids()
    new_rejected_rows: list = []
    total_saved = 0

    for dtype in DISASTER_TYPES:
        meta_dir  = METADATA_DIR / dtype
        frame_out = FRAMES_DIR / dtype

        if force and frame_out.exists():
            shutil.rmtree(frame_out)
            log.info(f"[{dtype}] Cleared existing frames (--force).")
        frame_out.mkdir(parents=True, exist_ok=True)

        if not meta_dir.exists():
            log.warning(f"[{dtype}] No metadata directory – skipping (run Stage 1 first).")
            continue

        video_ids = [jf.stem for jf in sorted(meta_dir.glob("*.info.json"))]
        pre_rejected = sum(1 for vid_id in video_ids if vid_id in rejected_ids)
        log.info(f"[{dtype}] {len(video_ids)} videos in metadata "
                 f"({pre_rejected} pre-rejected by Stage 2).")

        saved_embs: list = []
        saved_count = 0
        lap_thresh = LAPLACIAN_THRESHOLDS.get(dtype, LAPLACIAN_THRESHOLD)

        for vid_id in tqdm(video_ids, desc=f"{dtype} extract", unit="video"):
            if saved_count >= TARGET_FRAMES_PER_TYPE:
                break
            if vid_id in rejected_ids:
                continue

            # Resume: skip videos whose frames already exist
            if not force:
                sub_existing  = list((frame_out / vid_id).glob("*.jpg")) \
                                 if (frame_out / vid_id).is_dir() else []
                flat_existing = list(frame_out.glob(f"{vid_id}_f*.jpg"))
                if sub_existing or flat_existing:
                    saved_count += len(sub_existing or flat_existing)
                    continue

            # Use existing local copy if available (old pipeline videos); otherwise download
            local_video = VIDEOS_DIR / dtype / f"{vid_id}.mp4"
            use_local   = local_video.exists()
            active_video = local_video if use_local else TEMP_VIDEO

            if not use_local:
                url = f"https://www.youtube.com/watch?v={vid_id}"
                try:
                    subprocess.run([
                        "yt-dlp", url,
                        "--format",
                        "bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]"
                        "/bestvideo[height>=480][ext=mp4]+bestaudio[ext=m4a]/best",
                        "--merge-output-format", "mp4",
                        "--output", str(TEMP_VIDEO),
                        "--no-playlist", "--quiet", "--no-warnings",
                    ], check=False, timeout=300)
                except subprocess.TimeoutExpired:
                    log.warning(f"  [{vid_id}] Download timed out – skipping.")
                    continue
                if not TEMP_VIDEO.exists():
                    log.warning(f"  [{vid_id}] Download produced no file – skipping.")
                    continue
            else:
                log.debug(f"  [{vid_id}] Using local video file.")

            try:
                # Inline CLIP + Laplacian quality check
                reason = _check_video(active_video, clip_model, clip_preprocess,
                                      text_features, n_aerial, device, lap_thresh)
                if reason:
                    new_rejected_rows.append({"video_id": vid_id, "reason": reason, "type": dtype})
                    rejected_ids.add(vid_id)
                    log.info(f"  REJECT {vid_id}: {reason}")
                    continue

                # Extract frames
                frames = _extract_frames_hybrid(active_video)
                log.info(f"  {vid_id}: {len(frames)} candidate frames")

                for frame_idx, frame_bgr in frames:
                    if saved_count >= TARGET_FRAMES_PER_TYPE:
                        break
                    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    emb = _dino_embed(pil, dino, dino_transform, device).squeeze(0)
                    if saved_embs:
                        emb_mat = torch.stack(saved_embs)
                        sims    = emb_mat @ emb
                        if float(sims.max()) > DINO_SIM_THRESHOLD:
                            continue
                    fname    = f"{vid_id}_f{frame_idx:06d}.jpg"
                    out_path = frame_out / fname
                    cv2.imwrite(str(out_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_embs.append(emb)
                    saved_count += 1

            finally:
                # Only delete the temp file; never touch a permanent local video
                if not use_local and TEMP_VIDEO.exists():
                    TEMP_VIDEO.unlink()

        log.info(f"[{dtype}] Saved {saved_count} unique frames.")
        total_saved += saved_count

    # Append new Stage-3 rejections to rejected_videos.csv
    if new_rejected_rows:
        reject_csv = ROOT / "rejected_videos.csv"
        existing_rows: list = []
        if reject_csv.exists():
            with open(reject_csv, newline="") as f:
                existing_rows = list(csv.DictReader(f))
        all_rows = existing_rows + new_rejected_rows
        with open(reject_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "reason", "type"])
            writer.writeheader()
            writer.writerows(all_rows)
        log.info(f"  Appended {len(new_rejected_rows)} Stage-3 rejections to rejected_videos.csv.")

    log.info(f"Stage 3 complete. Total frames: {total_saved}")
    mark_complete(3)


def _extract_frames_hybrid(vpath: Path) -> list:
    """
    Return (frame_index, frame_bgr) pairs combining:
      - PySceneDetect scene-boundary frames
      - Uniform frames at 1 frame per 5 seconds
    Indices are merged and deduplicated before decoding.
    """
    import cv2
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector

    # ── Scene-change indices ──────────────────────────────────────────────────
    scene_indices: set = set()
    try:
        video_manager = VideoManager([str(vpath)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        video_manager.set_downscale_factor(2)
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        for start_tc, _ in scene_manager.get_scene_list():
            scene_indices.add(start_tc.get_frames())
        video_manager.release()
    except Exception as e:
        log.warning(f"  PySceneDetect error on {vpath.name}: {e}")

    # ── Uniform indices (1 frame every 5 s) ───────────────────────────────────
    cap   = cv2.VideoCapture(str(vpath))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps * 5))
    uniform_indices = set(range(0, total, step))

    # ── Decode merged set ─────────────────────────────────────────────────────
    all_indices = sorted(scene_indices | uniform_indices)
    frames = []
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))

    cap.release()
    return frames


def _dino_embed(pil_img, model, transform, device):
    import torch
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    return emb / emb.norm(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 – CLIP+GrabCut annotation + Roboflow upload
#
# Roboflow auto-label API is not available on the Core plan, so we:
#   1. Run CLIP to score which of the 10 classes are present in each frame
#   2. Run GrabCut for each detected class to produce a segmentation mask
#   3. Convert masks → YOLO segmentation format (.txt per image)
#   4. Upload image + annotation together via project.upload()
#      (duplicate images get their existing Roboflow ID back; annotation attaches)
# ─────────────────────────────────────────────────────────────────────────────
def stage4_auto_labeling():
    log.info("=" * 60)
    log.info("STAGE 4 – CLIP+k-means annotation → Roboflow upload")
    log.info("=" * 60)

    _ensure_packages(["tqdm", "roboflow", "torch", "Pillow", "opencv-python",
                      "git+https://github.com/openai/CLIP.git"])

    import cv2
    import numpy as np
    import torch
    import clip
    from PIL import Image
    from tqdm import tqdm
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        log.error("ROBOFLOW_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    # ── Roboflow connection ───────────────────────────────────────────────────
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("disasterview-seg-zqdc0")
    log.info("Connected to Roboflow project: disasterview-seg-zqdc0")

    # Class ID → name labelmap for Roboflow annotation upload
    labelmap = {str(i): LABEL_PROMPTS[i] for i in range(len(LABEL_PROMPTS))}

    # ── CLIP setup ────────────────────────────────────────────────────────────
    device = "cpu"
    log.info("Loading CLIP ViT-B/32 …")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(LABEL_PROMPTS).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    log.info("CLIP loaded.")

    PRESENCE_THRESHOLD = 0.05   # softmax prob; class included if above this
    TOP_K_CLASSES = 3           # annotate top-K classes per frame

    uploaded = 0
    annotated = 0
    skipped = 0
    failed = 0

    for dtype in DISASTER_TYPES:
        frame_dir = FRAMES_DIR / dtype
        if not frame_dir.exists():
            log.warning(f"[{dtype}] No frames directory – skipping.")
            continue

        frames = sorted(frame_dir.rglob("*.jpg")) + sorted(frame_dir.rglob("*.png"))
        log.info(f"[{dtype}] Processing {len(frames)} frames.")

        for fpath in tqdm(frames, desc=f"{dtype}", unit="frame"):
            # Resume: skip frames that already have a saved annotation
            ann_path = fpath.with_suffix(".txt")
            if ann_path.exists():
                skipped += 1
                continue

            try:
                ann_txt = _clip_kmeans_yolo(
                    fpath, clip_model, clip_preprocess, text_feats,
                    device, PRESENCE_THRESHOLD, TOP_K_CLASSES
                )

                if ann_txt:
                    ann_path.write_text(ann_txt)
                    try:
                        project.upload(
                            image_path=str(fpath),
                            annotation_path=str(ann_path),
                            annotation_labelmap=labelmap,
                            split="train",
                            batch_name=f"disasterview-{dtype}",
                            annotation_overwrite=True,
                        )
                        annotated += 1
                    except Exception as upload_err:
                        log.warning(f"  Upload failed {fpath.name}: {upload_err}")
                        # Keep the .txt so we can retry the upload separately
                else:
                    # No classes above threshold; upload image-only (idempotent)
                    project.upload_image(
                        image_path=str(fpath),
                        split="train",
                        batch_name=f"disasterview-{dtype}",
                    )

                uploaded += 1

            except Exception as e:
                log.warning(f"  Failed {fpath.name}: {e}")
                failed += 1

    log.info(f"Stage 4 complete. "
             f"Processed: {uploaded} | Annotated: {annotated} | "
             f"Skipped (already done): {skipped} | Failed: {failed}")
    mark_complete(4)


def _clip_kmeans_yolo(fpath: Path, clip_model, clip_preprocess, text_feats,
                      device: str, presence_threshold: float,
                      top_k: int = 3) -> str:
    """
    Generate YOLO-segmentation annotation using CLIP class scoring + k-means
    color clustering. ~0.15 s/frame on CPU (vs ~37 s for GrabCut).

    Approach:
      1. CLIP global scores → softmax → top-K classes present in frame
      2. Downsample image 25%, convert to LAB, run k-means (K=5, 10 iters)
      3. Assign each k-means cluster to the top-K ranked classes in order
      4. Upsample cluster mask → largest contour → approxPolyDP → YOLO seg line

    YOLO seg format per line: <class_id> <x1_norm> <y1_norm> …
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    pil    = Image.open(fpath).convert("RGB")
    img_np = np.array(pil)
    h, w   = img_np.shape[:2]

    # CLIP class-presence scores
    img_t = clip_preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ text_feats.T).squeeze(0).cpu().numpy()

    exp_s = np.exp(sims - sims.max())
    probs = exp_s / exp_s.sum()

    ranked = sorted(
        [(i, float(p)) for i, p in enumerate(probs) if p >= presence_threshold],
        key=lambda x: -x[1]
    )[:top_k]

    if not ranked:
        return ""

    # Downsample to 25% for fast k-means
    sw, sh = max(1, w // 4), max(1, h // 4)
    small  = cv2.resize(img_np, (sw, sh), interpolation=cv2.INTER_AREA)
    lab    = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    K        = min(5, len(ranked) + 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(lab, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(sh, sw)

    # Order clusters by size descending; assign to ranked classes in order
    cluster_order = sorted(range(K), key=lambda k: -np.sum(labels == k))

    lines = []
    for rank_idx, (class_id, _prob) in enumerate(ranked):
        if rank_idx >= len(cluster_order):
            break
        cluster_id = cluster_order[rank_idx]

        small_mask = ((labels == cluster_id).astype(np.uint8)) * 255
        mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            continue

        eps  = 0.02 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps, True).squeeze()
        if poly.ndim < 2 or len(poly) < 3:
            continue

        coords = " ".join(f"{pt[0] / w:.6f} {pt[1] / h:.6f}" for pt in poly)
        lines.append(f"{class_id} {coords}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 – Quality Verification
# ─────────────────────────────────────────────────────────────────────────────
def stage5_quality_verification():
    log.info("=" * 60)
    log.info("STAGE 5 – Quality Verification")
    log.info("=" * 60)

    _ensure_packages(["tqdm", "torch", "Pillow", "git+https://github.com/openai/CLIP.git"])

    import json
    import torch
    import clip
    from PIL import Image
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"CLIP verification device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(LABEL_PROMPTS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)

    quality_rows = []

    for dtype in DISASTER_TYPES:
        frame_dir  = FRAMES_DIR / dtype
        ann_file   = ANNOTATIONS_DIR / dtype / "annotations_coco.json"

        if not frame_dir.exists():
            log.warning(f"[{dtype}] No frames – skipping.")
            continue

        verified_out = VERIFIED_DIR / dtype
        review_out   = REVIEW_DIR / dtype
        verified_out.mkdir(parents=True, exist_ok=True)
        review_out.mkdir(parents=True, exist_ok=True)

        coco_annotations = {}
        if ann_file.exists():
            coco_data = json.loads(ann_file.read_text())
            for ann in coco_data.get("annotations", []):
                img_id = ann["image_id"]
                coco_annotations.setdefault(img_id, []).append(ann)
            img_id_map = {img["file_name"]: img["id"] for img in coco_data.get("images", [])}
        else:
            img_id_map = {}

        frames = sorted(frame_dir.rglob("*.jpg")) + sorted(frame_dir.rglob("*.png"))
        log.info(f"[{dtype}] Verifying {len(frames)} frames.")

        for fpath in tqdm(frames, desc=f"{dtype} verify", unit="frame"):
            pil = Image.open(fpath).convert("RGB")
            img_tensor = preprocess(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sims = (img_feat @ text_features.T).squeeze(0).cpu()

            # Use top-matching label similarity as confidence proxy
            max_sim, best_label_idx = sims.max(0)
            confidence = float(max_sim)
            best_label = LABEL_PROMPTS[int(best_label_idx)]

            flagged = confidence < CLIP_VERIFY_THRESHOLD
            dest_dir = review_out if flagged else verified_out
            shutil.copy2(fpath, dest_dir / fpath.name)

            img_id = img_id_map.get(fpath.name, -1)
            quality_rows.append({
                "file": fpath.name,
                "type": dtype,
                "best_label": best_label,
                "confidence": round(confidence, 4),
                "flagged": flagged,
                "img_id": img_id,
            })

    quality_csv = ROOT / "quality_report.csv"
    with open(quality_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "type", "best_label", "confidence", "flagged", "img_id"])
        writer.writeheader()
        writer.writerows(quality_rows)

    verified_count = sum(1 for r in quality_rows if not r["flagged"])
    flagged_count  = sum(1 for r in quality_rows if r["flagged"])
    log.info(f"Stage 5 complete. Verified: {verified_count} | Flagged: {flagged_count} → {quality_csv}")
    mark_complete(5)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 – Roboflow Upload
# ─────────────────────────────────────────────────────────────────────────────
def stage6_roboflow_upload():
    log.info("=" * 60)
    log.info("STAGE 6 – Generate Roboflow Dataset Version 2 (video-level 70/15/15)")
    log.info("=" * 60)

    _ensure_packages(["roboflow"])

    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        log.error("ROBOFLOW_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    manifest_path = ROOT / "split_manifest.json"
    if not manifest_path.exists():
        log.error("split_manifest.json not found – run reorganize_splits.py first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    for split_name in ["train", "val", "test"]:
        vids   = manifest["splits"][split_name]
        frames = sum(manifest["videos"][v]["frame_count"] for v in vids)
        log.info(f"  {split_name:5s}: {len(vids):3d} videos, {frames:5d} frames")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("disasterview-seg-zqdc0")
    log.info("Connected to Roboflow project: disasterview-seg-zqdc0")

    log.info("Generating dataset version 2 – raw frames, no augmentations …")
    version = project.generate_version(
        settings={
            "preprocessing": {
                "auto-orient": True,
            },
            "augmentation": {},
            "split": {
                "train":      70,
                "validation": 15,
                "test":       15,
            },
        }
    )

    log.info(f"Dataset version 2 generated: {version}")
    log.info("Note: video-level split assignments are in split_manifest.json. "
             "Roboflow's internal split is a random 70/15/15 used for its own UI; "
             "use the manifest as the authoritative split when training locally.")
    log.info("Stage 6 complete.")
    mark_complete(6)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 – HuggingFace Dataset Upload
# ─────────────────────────────────────────────────────────────────────────────
def stage7_huggingface_upload():
    log.info("=" * 60)
    log.info("STAGE 7 – HuggingFace Dataset Upload → mahergzani/disasterview")
    log.info("=" * 60)

    _ensure_packages(["huggingface_hub"])

    from huggingface_hub import HfApi, DatasetCard, DatasetCardData

    token = os.environ.get("HUGGINGFACE_TOKEN", "")
    if not token:
        log.error("HUGGINGFACE_TOKEN not set. Add it to .env and retry.")
        sys.exit(1)

    api = HfApi(token=token)
    repo_id = "mahergzani/disasterview"

    # Create repo (no-op if it already exists)
    log.info(f"Creating/verifying dataset repo: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    # Generate and upload dataset card (README.md)
    log.info("Generating dataset card …")
    card_content = """\
---
license: cc-by-4.0
task_categories:
  - image-segmentation
tags:
  - UAV
  - drone
  - disaster
  - aerial
  - segmentation
  - earthquake
  - flood
  - tornado
  - wildfire
size_categories:
  - 10K<n<100K
---

# DisasterView

DisasterView is a large-scale UAV/drone aerial semantic segmentation dataset covering four
natural disaster types: **earthquake**, **flood**, **tornado**, and **wildfire**. It contains
over 32,000 annotated frames extracted from 842 unique YouTube videos, with pixel-level
polygon annotations for 10 semantic classes. The dataset is designed to support automated
disaster-assessment models that aid first responders, with video-disjoint train/val/test
splits that prevent data leakage across partitions.

All annotations were generated by an automated pipeline: CLIP+Laplacian quality filtering →
PySceneDetect keyframe extraction → DINOv2 deduplication → CLIP+k-means segmentation →
CLIP confidence verification.

> **NeurIPS 2026 Datasets & Benchmarks Track submission**
> Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), Abu Dhabi

---

## Dataset Statistics

| Disaster Type | Videos | Frames |
|--------------|--------|--------|
| earthquake   | 80     | 2,903  |
| flood        | 208    | 7,722  |
| tornado      | 334    | 13,406 |
| wildfire     | 220    | 8,202  |
| **Total**    | **842**| **32,233** |

---

## Semantic Classes

| ID | Class | Description |
|----|-------|-------------|
| 0  | background | Sky, bare ground, and featureless surfaces |
| 1  | building_damaged | Collapsed, partially destroyed, or fire-damaged structures |
| 2  | building_intact | Standing, undamaged buildings and rooftops |
| 3  | debris_rubble | Loose rubble, wreckage, and scattered building materials |
| 4  | fire_smoke | Active flames and smoke plumes |
| 5  | road_blocked | Roads obstructed by debris, water, or damage |
| 6  | road_clear | Passable roads and open pathways |
| 7  | vegetation | Trees, grass, shrubs, and other plant cover |
| 8  | vehicle | Cars, trucks, emergency vehicles |
| 9  | water_flood | Flood water, inundated terrain |

---

## Splits

Splits are **video-disjoint**: frames from a given source video appear in exactly one
of train / val / test. This prevents data leakage from temporal correlation within
a video. The authoritative split assignments are in `split_manifest.json`.

```python
import json, pathlib

manifest = json.loads(pathlib.Path("split_manifest.json").read_text())
train_videos = manifest["splits"]["train"]   # list of video_ids
val_videos   = manifest["splits"]["val"]
test_videos  = manifest["splits"]["test"]
```

**Do not use Roboflow's built-in split** (it assigns frames randomly without video-level
grouping, causing leakage).

---

## File Structure

```
disasterview-raw/          # JPEG frames + YOLO-seg .txt annotations
  earthquake/<video_id>/   # one subdirectory per source video
  flood/<video_id>/
  tornado/<video_id>/
  wildfire/<video_id>/
disasterview-coco/         # COCO segmentation format
split_manifest.json        # authoritative video-disjoint train/val/test split
video_provenance.csv       # source metadata for all 1,618 candidate videos
datasheet.md               # Gebru et al. datasheet
annotation_guide.md        # annotation methodology and quality criteria
metadata.json              # Croissant metadata (schema.org + MLCommons)
LICENSE.txt                # CC BY 4.0
```

---

## Loading Example

```python
import json, pathlib
from PIL import Image

root = pathlib.Path("disasterview-raw")
manifest = json.loads(pathlib.Path("split_manifest.json").read_text())

for video_id in manifest["splits"]["train"]:
    dtype = manifest["videos"][video_id]["disaster_type"]
    video_dir = root / dtype / video_id
    for img_path in sorted(video_dir.glob("*.jpg")):
        ann_path = img_path.with_suffix(".txt")
        image = Image.open(img_path)
        annotations = ann_path.read_text() if ann_path.exists() else ""
        # annotations: one line per segment → "<class_id> x1 y1 x2 y2 ..."
```

---

## Citation

```bibtex
@dataset{disasterview2026,
  title     = {DisasterView: A Large-Scale UAV Aerial Segmentation Dataset for Natural Disasters},
  author    = {Guizani, Maher and others},
  year      = {2026},
  publisher = {Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)},
  url       = {https://huggingface.co/datasets/mahergzani/disasterview},
  note      = {NeurIPS 2026 Datasets and Benchmarks Track submission}
}
```

---

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — source videos remain subject
to YouTube Terms of Service and individual creator copyrights. See `video_provenance.csv`
for per-video attribution.
"""

    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    log.info("Dataset card uploaded.")

    # Files to upload directly
    single_files = [
        (ROOT / "split_manifest.json",                    "split_manifest.json"),
        (ROOT / "video_provenance.csv",                   "video_provenance.csv"),
        (NEURIPS_DIR / "datasheet.md",                    "datasheet.md"),
        (NEURIPS_DIR / "annotation_guide.md",             "annotation_guide.md"),
        (NEURIPS_DIR / "LICENSE.txt",                     "LICENSE.txt"),
        (NEURIPS_DIR / "metadata.json",                   "metadata.json"),
    ]

    for local_path, repo_path in single_files:
        if not local_path.exists():
            log.warning(f"  Skipping {repo_path} – file not found at {local_path}")
            continue
        log.info(f"  Uploading {repo_path} …")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {repo_path}",
        )

    # Folders to upload recursively
    folders = [
        (ROOT / "exports" / "disasterview-coco", "disasterview-coco"),
        (ROOT / "exports" / "disasterview-raw",  "disasterview-raw"),
    ]

    for local_dir, repo_dir in folders:
        if not local_dir.exists():
            log.warning(f"  Skipping {repo_dir}/ – directory not found at {local_dir}")
            continue
        file_count = sum(1 for _ in local_dir.rglob("*") if _.is_file())
        log.info(f"  Uploading {repo_dir}/ ({file_count} files) …")
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=repo_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {repo_dir}/",
        )
        log.info(f"  {repo_dir}/ done.")

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    log.info(f"Stage 7 complete.")
    log.info(f"Dataset published at: {dataset_url}")
    print(f"\n{'=' * 60}")
    print(f"  DisasterView dataset live at:")
    print(f"  {dataset_url}")
    print(f"{'=' * 60}\n")
    mark_complete(7)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_packages(packages: list):
    """Install packages that are not already importable."""
    import importlib
    to_install = []
    for pkg in packages:
        # Strip version specifiers and git+... prefixes for the import check
        mod = pkg.split(">=")[0].split("==")[0].split("[")[0]
        mod = mod.split("/")[-1].replace("-", "_").lower()
        if mod.startswith("git+"):
            to_install.append(pkg)
            continue
        try:
            importlib.import_module(mod)
        except ImportError:
            to_install.append(pkg)

    if to_install:
        log.info(f"Installing: {to_install}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + to_install)


def _download_file(url: str, dest: Path):
    import urllib.request
    log.info(f"  Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
STAGE_FNS = {
    1: stage1_youtube_collection,
    2: stage2_quality_filtering,
    3: stage3_frame_extraction,
    4: stage4_auto_labeling,
    5: stage5_quality_verification,
    6: stage6_roboflow_upload,
    7: stage7_huggingface_upload,
}


def main():
    parser = argparse.ArgumentParser(description="DisasterView Dataset Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=int, choices=range(1, 8),
                       help="Run a single stage (1-7)")
    group.add_argument("--all", action="store_true",
                       help="Run all stages sequentially (1-7)")
    parser.add_argument("--type", dest="type_filter", choices=DISASTER_TYPES, default=None,
                        help="Limit stage 1 or 2 to a single disaster type "
                             "(stage 1 + earthquake triggers the boost-query set)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run a stage even if already marked complete; "
                             "for stage 3 this also clears existing extracted frames")
    args = parser.parse_args()

    if args.type_filter and args.all:
        parser.error("--type cannot be combined with --all")

    for d in [VIDEOS_DIR, FRAMES_DIR, ANNOTATIONS_DIR, REVIEW_DIR, VERIFIED_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    stages_to_run = list(range(1, 8)) if args.all else [args.stage]
    status = load_status()

    # Stages that accept a type_filter argument
    FILTERABLE = {1, 2}

    for s in stages_to_run:
        if args.all and not args.force and status.get(str(s), {}).get("complete"):
            log.info(f"Stage {s} already complete – skipping.")
            continue
        log.info(f"\n{'#' * 60}\n# Running Stage {s}\n{'#' * 60}")
        t0 = time.time()
        if s == 3:
            stage3_frame_extraction(force=args.force)
        elif s in FILTERABLE and args.type_filter:
            STAGE_FNS[s](type_filter=args.type_filter)
        else:
            STAGE_FNS[s]()
        elapsed = time.time() - t0
        log.info(f"Stage {s} finished in {elapsed:.1f}s")

        # After stage 4, reorganize frames into per-video subdirs and refresh split manifest
        if s == 4 and args.all:
            log.info(f"\n{'#' * 60}\n# Running reorganize_splits\n{'#' * 60}")
            t0 = time.time()
            subprocess.check_call([sys.executable, str(ROOT / "reorganize_splits.py")])
            log.info(f"reorganize_splits finished in {time.time() - t0:.1f}s")

    print("\nPipeline status:")
    status = load_status()
    for s in range(1, 8):
        info = status.get(str(s), {})
        state = "✓ complete" if info.get("complete") else "○ pending"
        ts    = info.get("timestamp") or ""
        print(f"  Stage {s}: {state}  {ts}")


if __name__ == "__main__":
    main()
