# DisasterView — Dataset Hosting Checklist

---

## HuggingFace Dataset Repository Setup

- [ ] Create HuggingFace account / organization at https://huggingface.co
- [ ] Create new dataset repository: `mbzuai/disasterview`
  ```bash
  pip install huggingface_hub
  huggingface-cli login
  huggingface-cli repo create disasterview --type dataset --organization mbzuai
  ```
- [ ] Upload dataset files:
  ```bash
  # Install Git LFS (required for large files)
  git lfs install

  git clone https://huggingface.co/datasets/mbzuai/disasterview
  cd disasterview

  # Copy files
  cp ~/Documents/Uni/MBZUAI/DisasterView/disasterview-v3.zip .
  cp ~/Documents/Uni/MBZUAI/DisasterView/split_manifest.json .
  cp ~/Documents/Uni/MBZUAI/DisasterView/video_provenance.csv .
  cp ~/Documents/Uni/MBZUAI/DisasterView/neurips_submission/metadata.json .
  cp ~/Documents/Uni/MBZUAI/DisasterView/neurips_submission/LICENSE.txt .
  cp ~/Documents/Uni/MBZUAI/DisasterView/neurips_submission/datasheet.md .

  git add .
  git commit -m "Initial DisasterView v3.0 release"
  git push
  ```
- [ ] Set repository visibility to **Public**
- [ ] Add dataset card (`README.md`) with:
  - Dataset summary, task type (semantic segmentation), language (none)
  - License: `cc-by-4.0`
  - Size category: `10B<n<100B` tokens
  - Tags: `aerial`, `UAV`, `disaster`, `segmentation`, `earthquake`, `flood`, `tornado`, `wildfire`
- [ ] Upload Croissant metadata (`metadata.json`) — use HuggingFace's metadata editor
- [ ] Verify dataset appears at: `https://huggingface.co/datasets/mbzuai/disasterview`

---

## Public Access Without Personal Request

NeurIPS D&B Track requires datasets to be **publicly accessible without requesting access**.

- [ ] Confirm HuggingFace repo is set to Public (not gated)
- [ ] Confirm no "Request Access" button is enabled
- [ ] Test anonymous access:
  ```python
  from datasets import load_dataset
  ds = load_dataset("mbzuai/disasterview")
  ```
- [ ] Verify direct download URL works (no login required):
  ```
  https://huggingface.co/datasets/mbzuai/disasterview/resolve/main/split_manifest.json
  ```
- [ ] Add a download script (`download_disasterview.py`) to the repo for ease of use

---

## Croissant Validation Steps

NeurIPS 2026 requires Croissant metadata to be validated.

- [ ] Install the Croissant validator:
  ```bash
  pip install mlcroissant
  ```
- [ ] Validate metadata.json:
  ```bash
  python -m mlcroissant validate --jsonld neurips_submission/metadata.json
  ```
- [ ] Fix any validation errors (common issues: missing `@id` fields, incorrect `@type`)
- [ ] Re-validate after fixes
- [ ] Upload validated `metadata.json` to HuggingFace repository root

---

## Long-Term Preservation Plan

| Platform         | Role                                      | Timeline         |
|-----------------|-------------------------------------------|------------------|
| HuggingFace      | Primary distribution, versioned releases  | Ongoing          |
| Roboflow         | Project `disasterview-seg-zqdc0`, workspace `mahair` | Ongoing |
| Zenodo           | Immutable DOI-assigned archive (backup)   | Before camera-ready |
| MBZUAI servers   | Raw videos + full pipeline                | 5+ years         |

### Zenodo Backup (Recommended)
```bash
# Install zenodo_upload or use web UI at https://zenodo.org
# Upload disasterview-v3.zip + metadata.json
# This creates a permanent DOI: 10.5281/zenodo.XXXXXXX
# Add DOI to paper and HuggingFace README
```

### Commitment
The MBZUAI research team commits to:
- Maintaining the HuggingFace repository for at least 5 years
- Responding to bug reports via the repository issue tracker
- Releasing future dataset versions under the same CC BY 4.0 license

---

## Pre-Release Checks

- [ ] All files upload successfully to HuggingFace
- [ ] `split_manifest.json` is accessible without login
- [ ] `video_provenance.csv` is accessible without login
- [ ] README/data card describes the dataset accurately
- [ ] Croissant metadata validates without errors
- [ ] License file is present and correct
- [ ] Dataset is indexed by HuggingFace search within 24 hours of publication
