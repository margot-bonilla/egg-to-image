# Neural Interfaces 2026 - Track 1: EEG-to-Image

Official Codabench competition: [Neural Interfaces 2026 - Track 1: EEG-to-Image](https://www.codabench.org/competitions/17974/)  
Challenge Website: [neural-interfaces26.github.io](https://neural-interfaces26.github.io/tracks.html#track-1)  
Workshop: NeurIPS 2026 Brain & Body Workshop (Sydney)

---

## 🎯 Task Overview
Given a single multichannel EEG epoch recorded while a human subject views a natural image, decode the image by predicting its **1536-dimensional visual embedding** in the frozen `facebook/dinov2-giant` feature space.

- **Input:** Single EEG epoch $X \in \mathbb{R}^{B \times C \times T}$ ($C$ channels, $T$ time points).
- **Target / Output:** 1536-dimensional feature vector $\hat{y} \in \mathbb{R}^{B \times 1536}$ aligned with DINOv2-giant.
- **Evaluation Metric:** **Top-5 Retrieval Accuracy** against the complete held-out image candidate gallery. A prediction is correct if the true viewed image is among the 5 nearest candidates in the embedding space (using cosine similarity or Euclidean distance).
- **Generalization Shift:** Evaluated on **unseen participants** and **unseen images** (cross-stimulus & cross-subject).

---

## 📂 Datasets

1. **Alljoined-1.6M (`Xu2025Alljoined`)**:
   - 20 participants, 130 hours, 32-channel Emotiv EEG, 256 Hz.
   - **Crucial:** The hidden test cohort (11 unseen subjects) is collected with the *exact same 32-channel Emotiv hardware and protocol*. This is your primary target distribution!
2. **THINGS-EEG2 (`Gifford2022Large`)**:
   - 10 participants, 87 hours, 63 EEG channels, 1000 Hz.
   - Standard benchmark dataset in NeuralBench.
3. **THINGS-EEG1 (`Grootswagers2022Human`)**:
   - 50 participants, 46 hours, 63/128 EEG channels, 1000 Hz.
4. **Alljoined-1 (`Xu2024Alljoined`)**:
   - 8 participants, 64 channels, 512 Hz.

---

## 📦 Submission Format & Platform Rules

Submissions on Codabench are **inference-only**:
- You submit a ZIP file containing:
  - `submission.py` defining a Benchopt solver subclassing `CompetSolver`.
  - Model weight checkpoint (e.g. `weights.pt`).
- **Resource limit:** Full test pass must run in under **60 minutes** on 1 GPU (H100/H200).
- **Warm-up Phase:** Up to 5 submissions per day.
- **Final Phase:** 1 submission per day (top 3 undergo reproducibility audit from code & config).

---

## 🏗️ Repository Structure

```
egg-to-image/
├── README.md
├── requirements.txt
├── submission/
│   ├── submission.py        # Codabench / Benchopt entrypoint
│   └── weights.pt           # Exported model weights
├── src/
│   ├── models/
│   │   └── eeg_encoder.py   # EEG feature extractor & projection head
│   ├── losses/
│   │   └── retrieval_loss.py# InfoNCE / Cosine alignment loss
│   └── metrics.py           # Top-1, Top-5 retrieval accuracy
├── scripts/
│   └── smoke_test.py        # Offline verification test with dummy data
└── configs/
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify with Smoke Test
```bash
python scripts/smoke_test.py
```

### 3. Local Benchopt Test (if using the competition harness)
```bash
benchopt install tracks/eeg_to_image
benchopt run tracks/eeg_to_image -d Simulated
```
