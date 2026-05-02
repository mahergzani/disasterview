# NeurIPS 2026 Datasets & Benchmarks Track — Submission Checklist

*Track all requirements for DisasterView NeurIPS 2026 submission.*
*Deadlines: Abstract May 4 · Full paper + dataset + code May 6 · Camera-ready TBD*

---

## ⏰ Deadlines

| Milestone                                  | Deadline     | Status |
|-------------------------------------------|--------------|--------|
| Abstract submission (OpenReview)          | May 4, 2026  | [ ]    |
| Full paper + supplementary + dataset URL  | May 6, 2026  | [ ]    |
| Dataset publicly accessible online        | May 6, 2026  | [ ]    |
| Croissant metadata submitted              | May 6, 2026  | [ ]    |
| Baseline code publicly available          | May 6, 2026  | [ ]    |
| Rebuttal period                           | TBD          | [ ]    |
| Camera-ready                              | TBD          | [ ]    |

---

## 📄 Paper Formatting

- [ ] Paper uses `neurips_2026.sty` style file (download from NeurIPS website)
- [ ] Paper is max 9 pages (content) + unlimited references + supplementary
- [ ] Author names/affiliations anonymized for blind review
- [ ] Abstract is ≤ 250 words
- [ ] All figures have captions
- [ ] All tables are readable at 100% zoom
- [ ] Paper includes dataset URL (or placeholder if not yet live)
- [ ] Paper includes Croissant metadata URL or states it is in supplementary

---

## 👤 OpenReview Profiles

- [ ] All authors have OpenReview accounts at https://openreview.net
- [ ] All author profiles have:
  - [ ] Full name matching the paper
  - [ ] Institutional email verified
  - [ ] Publication history (if applicable)
  - [ ] DBLP / Semantic Scholar link (recommended)
- [ ] Paper uploaded to OpenReview with all author IDs linked

---

## 🗄️ Dataset Requirements

- [ ] Dataset is **publicly accessible** without requesting access
  - URL: `https://huggingface.co/datasets/mbzuai/disasterview`
- [ ] Dataset accessible without personal request (no gating)
- [ ] Persistent URL (HuggingFace or Zenodo with DOI)
- [ ] Dataset has version number (v3.0)
- [ ] Dataset has license specified (CC BY 4.0)
- [ ] `video_provenance.csv` included (source video attribution)

### Croissant Metadata

- [ ] `metadata.json` in Croissant format created ✓ (`neurips_submission/metadata.json`)
- [ ] Validated with `mlcroissant` validator
- [ ] Uploaded to HuggingFace repository
- [ ] URL to Croissant file included in paper or supplementary

---

## 📋 Documentation

- [ ] Datasheet (Gebru et al. 2018) ✓ (`neurips_submission/datasheet.md`)
  - [ ] Converted to PDF for supplementary upload
- [ ] Data card ✓ (`neurips_submission/data_card.md`)
- [ ] Annotation guide ✓ (`neurips_submission/annotation_guide.md`)
- [ ] Statistics report ✓ (`neurips_submission/statistics.md`)
- [ ] LICENSE.txt ✓ (`neurips_submission/LICENSE.txt`)

---

## 💻 Code Requirements

- [ ] Baseline code is **publicly available** (GitHub or HuggingFace)
- [ ] Baseline code includes:
  - [ ] `requirements.txt` or `environment.yml`
  - [ ] Dataset loading with video-disjoint split (using `split_manifest.json`)
  - [ ] SegFormer-B0 training script
  - [ ] Evaluation script with per-class mIoU
  - [ ] README with reproduction instructions ✓ (`neurips_submission/baselines/README.md`)
- [ ] Code runs end-to-end on a single GPU in < 24 hours
- [ ] Results in paper reproducible from code

---

## ⚖️ Ethics and Responsible Use

- [ ] NeurIPS ethics checklist answered in paper (Appendix)
  - [ ] Dataset contains no personally identifiable information
  - [ ] No faces annotated or identified
  - [ ] Source video copyright disclosed (CC BY 4.0 for annotations; YouTube ToS for videos)
  - [ ] Potential misuse (surveillance) acknowledged and mitigated
  - [ ] Geographic/demographic bias disclosed
  - [ ] Annotation limitations (automated, no manual review) disclosed
- [ ] Vulnerable populations: dataset may depict people in disaster scenarios — no individuals identified
- [ ] No dual-use risks beyond standard computer vision limitations

---

## 🔍 Final Pre-Submission Checks

- [ ] Download and verify dataset from public URL
- [ ] Run baseline code from scratch on clean environment
- [ ] Confirm mIoU numbers in paper match reproduced results
- [ ] Confirm all supplementary files are included in OpenReview upload:
  - [ ] `datasheet.pdf`
  - [ ] `metadata.json`
  - [ ] `annotation_guide.pdf`
  - [ ] `statistics.pdf`
  - [ ] `baselines/` directory
- [ ] Proofread paper abstract and introduction
- [ ] Spell-check all files
- [ ] All co-authors have reviewed the final submission

---

## 📬 Supplementary Material Upload to OpenReview

Upload `neurips_submission.zip` as supplementary. It contains:
```
neurips_submission/
├── video_provenance.csv
├── datasheet.md
├── metadata.json          ← Croissant
├── data_card.md
├── LICENSE.txt
├── statistics.md
├── annotation_guide.md
├── hosting_checklist.md
├── submission_checklist.md
└── baselines/
    └── README.md
```
