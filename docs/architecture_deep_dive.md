# Architecture deep dive: the model, the recipe, the evidence

This is the reference for teaching the project. It goes layer by layer through the model,
derives the loss, explains the evaluation, lists every experiment with its numbers, and maps
each idea to the paper it comes from.

- Start with [how_it_works.md](how_it_works.md) for the overview (task, data, big picture).
  This document assumes it and goes deeper.
- Code references are to commit `0e9ff1f` (2026-10-20) on branch `ws-m1-infra` (§9: `d3ed4ea`). Function and
  class names are stable; line numbers drift, so search for the name.
- Every number below comes from a file in the repo (`RESULTS.md`, `logs/`, `submissions/*/run_config*.json`)
  or from a script run on the model. Anything that couldn't be checked is marked **(unverified)**.

Contents
1. [The encoder, layer by layer](#1-the-encoder-layer-by-layer)
2. [Where the encoder comes from (lineage and history)](#2-where-the-encoder-comes-from)
3. [The loss and the training recipe](#3-the-loss-and-the-training-recipe)
4. [Evaluation in detail](#4-evaluation-in-detail)
5. [Every experiment and its result](#5-every-experiment-and-its-result)
6. [Submission and inference path](#6-submission-and-inference-path)
7. [Paper map and bibliography](#7-paper-map-and-bibliography)
8. [Teaching kit](#8-teaching-kit)
9. [Next: new people and new headsets (in progress)](#9-next-new-people-and-new-headsets-in-progress)

---

## 1. The encoder, layer by layer

`EEGToImageEncoder` in `submission/submission.py`. One input window is a matrix of
**63 electrodes × 120 time samples** (1 s at 120 Hz, from −0.2 s to +0.8 s around image onset).
The output is one **unit-length 1536-d vector** in DINOv2-giant's embedding space.

### 1.1 The whole network at a glance

```mermaid
flowchart TB
    IN["EEG window"] -->|"(B, 63, 120)"| U["unsqueeze: treat it as a 1-channel image"]
    U -->|"(B, 1, 63, 120)"| TC["Temporal conv 1→32, kernel (1, 31), pad 15<br/>+ BatchNorm"]
    TC -->|"(B, 32, 63, 120)"| SC["Spatial conv 32→64, kernel (63, 1)<br/>+ BatchNorm + ELU"]
    SC -->|"(B, 64, 1, 120)"| P1["AvgPool (1, 4) + Dropout 0.25"]
    P1 -->|"(B, 64, 1, 30)"| DW["Depthwise conv 64→64, kernel (1, 15), groups 64"]
    DW -->|"(B, 64, 1, 30)"| PW["Pointwise conv 64→128, kernel (1, 1)<br/>+ BatchNorm + ELU"]
    PW -->|"(B, 128, 1, 30)"| P2["AvgPool (1, 4) + Dropout 0.25"]
    P2 -->|"(B, 128, 1, 7)"| FL["Flatten"]
    FL -->|"(B, 896)"| H1["Linear 896→512 + BatchNorm + GELU + Dropout"]
    H1 -->|"(B, 512)"| H2["Linear 512→1536"]
    H2 -->|"(B, 1536)"| N["L2 normalise"]
    N -->|"(B, 1536), length 1"| OUT["Predicted image embedding"]
```

### 1.2 Layer table (computed, not estimated)

Shapes and parameter counts were produced by running the class on a `(2, 63, 120)` input on
CPU (script in exercise 1, §8.3).

| # | Module (exact PyTorch) | Shape in → out | Parameters |
|---|---|---|---|
| 1 | `Conv2d(1, 32, kernel_size=(1, 31), padding=(0, 15), bias=False)` | (B,1,63,120) → (B,32,63,120) | 992 |
| 2 | `BatchNorm2d(32)` | same | 64 |
| 3 | `Conv2d(32, 64, kernel_size=(63, 1), groups=1, bias=False)` | (B,32,63,120) → (B,64,1,120) | **129,024** |
| 4 | `BatchNorm2d(64)` · `ELU()` | same | 128 |
| 5 | `AvgPool2d((1, 4), stride=(1, 4))` · `Dropout(0.25)` | (B,64,1,120) → (B,64,1,30) | 0 |
| 6 | `Conv2d(64, 64, kernel_size=(1, 15), padding=(0, 7), groups=64, bias=False)` | (B,64,1,30) → same | 960 |
| 7 | `Conv2d(64, 128, kernel_size=(1, 1), bias=False)` | (B,64,1,30) → (B,128,1,30) | 8,192 |
| 8 | `BatchNorm2d(128)` · `ELU()` | same | 256 |
| 9 | `AvgPool2d((1, 4), stride=(1, 4))` · `Dropout(0.25)` | (B,128,1,30) → (B,128,1,7) | 0 |
| 10 | `flatten` | (B,128,1,7) → (B,896) | 0 |
| 11 | `Linear(896, 512)` | (B,896) → (B,512) | **459,264** |
| 12 | `BatchNorm1d(512)` · `GELU()` · `Dropout(0.25)` | same | 1,024 |
| 13 | `Linear(512, 1536)` | (B,512) → (B,1536) | **787,968** |
| 14 | `F.normalize(p=2, dim=-1)` | same | 0 |
| | **Total** | | **1,387,872** |

**Where the parameters are:** the convolutional "feature extractor" (layers 1–9) has 139,616
parameters (10%); the projection head (11–13) has 1,248,256 (90%). The network is mostly a
mapping from 896 EEG features to DINOv2 space. The single biggest conv layer is the spatial
one (129k), because every output filter sees all 32 temporal maps on all 63 electrodes.

Size variants from #27 (`temporal_filters`, `spatial_filters`, `sep_filters`, `hidden_dim`,
`head_blocks`), also computed: wider convs 64/128/256 → 2.26M; head 1024 + 1 residual block →
5.00M; both → 6.33M.

### 1.3 What each layer does, and why

**Input reshaping.** EEG is not an image: one axis is space (electrodes), the other is time.
Treating it as a 1-channel image `(1, 63, 120)` lets us use 2-D convolutions whose kernels
are 1-D in one axis: `(1, k)` kernels act **along time only**, `(63, 1)` kernels **across
electrodes only**. This "factorised" spatio-temporal design is the core idea of
ShallowConvNet / DeepConvNet (Schirrmeister et al., 2017) and EEGNet (Lawhern et al., 2018).

**1–2. Temporal convolution (32 filters × 31 taps) + BatchNorm.**
- Each filter is a **learned FIR filter** applied identically to every electrode (the kernel is
  `(1, 31)`, so it slides along time and is shared across rows).
- 31 taps at 120 Hz span **258 ms**. A filter of that length can isolate frequency bands about
  120/31 ≈ 4 Hz wide anywhere between 0 and the 60 Hz Nyquist limit: theta (4–8 Hz), alpha
  (8–13 Hz), beta, or act as a matched filter for the shape of an evoked response.
- `padding=15` keeps the 120 samples ("same" convolution). The first and last 15 samples are
  computed against zero padding, so the window edges are slightly less reliable.
- `bias=False` because the BatchNorm right after has its own shift.
- **BatchNorm2d** (Ioffe & Szegedy, 2015) standardises each filter's output over the batch,
  electrodes and time, which keeps training stable at a high learning rate.
- Lineage: EEGNet uses 8 filters of half the sampling rate (64 taps at 128 Hz ≈ 500 ms);
  ShallowConvNet uses 40 filters of 25 taps at 250 Hz (100 ms). Ours sits in between, with
  more filters.

**3–4. Spatial convolution (64 filters spanning all 63 electrodes) + BatchNorm + ELU.**
- Kernel `(63, 1)`: each output is a **weighted sum over all electrodes** at one time point.
  This is a learned **spatial filter**, the deep-learning version of classical techniques like
  Common Spatial Patterns (CSP), beamformers, or picking an ICA component: it can, for example,
  weight the occipital electrodes positively and subtract a frontal reference.
- `groups=1` means each of the 64 spatial filters mixes **all 32 temporal maps**: it combines
  frequency bands and electrodes jointly (32 × 63 = 2,016 weights per filter). EEGNet instead
  uses a **depthwise** spatial conv (each spatial filter sees only one temporal map, `D = 2`
  spatial filters per temporal filter, 63 weights each, with a max-norm constraint). Ours has
  many more parameters here (129k vs EEGNet's 1,008 at the same input size).
  *Note:* the original scaffold's comment calls this layer "depthwise" but the code has always
  used `groups=1` (commit `db3fb74`); the code is what ran in every experiment.
- The kernel height is the channel count, so **the layer is tied to 63 electrodes in a fixed
  order**. That's why a THINGS-EEG2 model can't run on the final phase's 32 Emotiv channels
  without retraining (see issue #21 and the planned position-based spatial attention).
- **ELU** (exponential linear unit, Clevert et al., 2016) is smooth for negative inputs;
  EEGNet and ShallowConvNet-family models use it.

**5. Average pooling (×4) + Dropout.**
- 120 → 30 time steps: an average over 4 samples is a crude **low-pass filter followed by
  downsampling** to 30 Hz (33 ms per step). Evoked visual responses are slow (most power below
  ~20 Hz), so little is lost and later layers become cheaper and see longer stretches of time.
- **Dropout 0.25** randomly zeroes 25% of activations during training (Srivastava et al.,
  2014). EEGNet's paper recommends 0.25 for cross-subject training (0.5 within-subject)
  **(from memory of the EEGNet paper; braindecode's default is 0.25)**.

**6–8. Depthwise-separable convolution + BatchNorm + ELU.**
- **Depthwise** (layer 6): each of the 64 maps gets its **own** 15-tap temporal filter
  (`groups=64`, 960 weights). At 30 Hz, 15 taps span **500 ms**: it summarises how each
  spatial component evolves over half a second.
- **Pointwise** (layer 7): a 1×1 conv that mixes the 64 maps into 128 (8,192 weights).
- Splitting "filter in time" from "mix the maps" is a **depthwise-separable convolution**
  (Chollet, Xception, CVPR 2017), adopted by EEGNet to use far fewer parameters than a full
  conv (a full 64→128 conv with 15 taps would need 122,880 weights instead of 9,152).

**9. Average pooling (×4) + Dropout.** 30 → 7 time steps (30/4 = 7.5, floored: the last two
30 Hz steps are not pooled, though their information still reaches step 6 through the
depthwise kernel). Each final step represents 16 input samples = **133 ms**.

**10. Flatten → 896 features.** 128 maps × 7 time steps. Flattening (rather than averaging
over time) keeps **when** each feature occurred: the next layer has separate weights for
"component 12 at 100–233 ms" and "component 12 at 500–633 ms". That matters because the
timing of visual responses carries information. It is also our best explanation for why
random time-shift augmentation hurt (R4, §5).

**Receptive field of the final features.** Following the layers back to the input:
temporal conv (31) → pool (stride 4) → depthwise conv (15 at stride 4) → pool (stride 16).
Each of the 7 final time steps sees **102 input samples = 850 ms**, and consecutive steps are
**133 ms apart**. Step 0 covers roughly −200 → +283 ms around image onset; step 6 roughly
+242 → +792 ms. So every final feature integrates most of the window, with a different
centre of gravity: early steps emphasise the first visual responses (~100–200 ms), late
steps the later, more semantic activity.

**11–13. Projection head.** `Linear(896→512) → BatchNorm1d → GELU → Dropout → Linear(512→1536)`.
- A two-layer MLP that maps EEG features into DINOv2's 1536-d space. Using an MLP
  "projector" between the encoder and the contrastive loss is the standard recipe in
  contrastive learning (CLIP uses linear projections; SimCLR showed a non-linear head helps).
- **GELU** (Hendrycks & Gimpel, 2016) is a smooth ReLU used in transformers.
- It's 90% of the model. The NeuralBench EEGNet baseline instead reads 48 features (16 maps ×
  3 time steps) straight into 1536 outputs with a single linear layer; that "bigger head" is
  one of the two changes `RESULTS.md` credits for our jump from 0.30 to 0.54 (§2.2).
- Optional (#27): `head_blocks` adds residual blocks `x + Dropout(Linear(GELU(x)))` on the
  1536-d output followed by `LayerNorm`, as in ATM's projector (Li et al., 2024). Off by
  default; R13/R14 in the queue test it.

**14. L2 normalisation.** Divides the output by its length so it lies on the unit sphere.
The metric ranks by **cosine similarity**, which only depends on direction, and the loss
normalises the targets too, so predictions and targets are compared purely by angle.

---

## 2. Where the encoder comes from

### 2.1 Family tree

```mermaid
flowchart LR
    FBCSP["Filter-bank CSP<br/>(classical BCI:<br/>band-pass, then spatial filters)"] --> SH["ShallowConvNet / DeepConvNet<br/>Schirrmeister et al. 2017<br/>temporal conv → spatial conv"]
    SH --> EEGNET["EEGNet<br/>Lawhern et al. 2018<br/>+ depthwise spatial conv<br/>+ separable conv, ELU, dropout"]
    XC["Xception<br/>Chollet 2017<br/>depthwise-separable conv"] --> EEGNET
    EEGNET --> NB["NeuralBench EEGNet baseline<br/>77k params, linear readout<br/>our 0.29"]
    EEGNET --> OURS["Our EEGToImageEncoder<br/>full spatial conv, wider,<br/>512-d MLP head, 1.39M params<br/>our 0.54 → 0.74"]
    CLIP["CLIP<br/>Radford et al. 2021<br/>contrastive loss, projector,<br/>learnable temperature"] --> OURS
    SH --> NICE["NICE<br/>Song et al. 2024<br/>ShallowNet-style patch encoder<br/>+ residual projector"]
    NICE --> ATM["ATM<br/>Li et al. 2024<br/>channel attention + conv<br/>+ residual projector"]
    ATM -.->|"head_blocks option, issue 27"| OURS
```

- **Filter-bank CSP** is the classical pipeline for EEG decoding: band-pass filter the signal
  into several frequency bands, learn spatial filters per band, then classify the log-power.
- **ShallowConvNet** (Schirrmeister et al., 2017) turned that into a network: temporal conv
  (40 filters × 25 taps) → spatial conv (40) → squaring → mean pooling (75, stride 15) → log
  → classifier, i.e. a learnable filter-bank CSP. **DeepConvNet** stacks four conv-pool blocks.
  (Defaults verified in braindecode's `ShallowFBCSPNet`: 40 / 25 / 40 / 75 / 15.)
- **EEGNet** (Lawhern et al., 2018) made it compact: temporal conv (F1 = 8 filters, 64 taps)
  → depthwise spatial conv (D = 2 per temporal filter, max-norm constrained) → BatchNorm, ELU,
  average pooling, dropout → separable conv (F2 = 16) → pooling → classifier. (Defaults
  verified in braindecode's `EEGNet`: F1 = 8, D = 2, F2 = F1·D, kernel 64, depthwise kernel 16,
  dropout 0.25.)
- **Our encoder** keeps EEGNet's block order and its separable conv, but uses 4× more
  temporal filters (32), a **full** (non-depthwise, unconstrained) spatial conv with 64
  filters, 128 separable maps, a shorter temporal kernel (31 taps at 120 Hz), and replaces
  the classifier with a 2-layer **projection head** into DINOv2 space trained with a CLIP loss.
- **NICE** (Song et al., ICLR 2024) and **ATM** (Li et al., NeurIPS 2024) are the THINGS-EEG2
  state of the art that issue #12 targets. From the papers and their public code, roughly
  **(unverified in detail; check the papers' architecture figures)**: NICE's encoder is a
  ShallowNet-style temporal conv → average pooling → spatial conv producing a short sequence
  of patch embeddings, then a projector `Linear → residual(GELU, Linear, Dropout) → LayerNorm`;
  NICE-GA adds graph attention over electrodes. ATM adds a channel-wise transformer embedding
  (with a subject token) before a similar conv block and residual projector. Our `head_blocks`
  option is that residual projector.

### 2.2 The true history (from git)

| Date | Commit | What happened |
|---|---|---|
| 2026-09-11 | `db3fb74` | Initial scaffold. `src/models/eeg_encoder.py` defines the architecture as **learning TODOs** "based on EEGNet principles": temporal conv (1, 31) × 32 → spatial conv (n_chans, 1) × 64 → separable conv → 512-d projection head → 1536, L2-normalised. The layer sizes we still use were fixed here. |
| 2026-09-13 | `7d7dc1e` | TODOs completed ("complete TODOs for setup"). |
| 2026-10-17/18 | `6896932`, `bf5bf17` | `submission/submission.py` made self-contained (torch only) and fixed for the Codabench contract; state_dict keys match `src/models/eeg_encoder.py`. |
| 2026-10-18 | `7b439e0` | **First real upload: NeuralBench's own EEGNet** (official template), 0.29. |
| 2026-10-19 | `da20e91`, `c652e3a` | Our encoder trained on THINGS-EEG2 with InfoNCE: **0.54** (held-out 42.15%). Seeded training and `weights_*.pt` ensembling added. |
| 2026-10-19 | `5facb9b` | Training options for the round-1 experiments (`learn_temperature`, `dedup`, `symmetric`, augmentation) and `--set` overrides. |
| 2026-10-19 | `1a8fc30`, `41b152e`, `2fb5a51` | Learnable temperature (+10 held-out), `--use-val` (+3.5). |
| 2026-10-20 | `e34d3f1` | Final 3-seed ensemble on all concepts: 73.80% test, **0.74** on Codabench. |
| 2026-10-20 | `18992e0`, `424c9b5` | Per-epoch held-out scoring + early stopping (#15); repetition averaging (#17). |
| 2026-10-20 | `e893e30`, `c5881bd`, `48925dd` | Schedule / weight decay / dropout settings (#18); encoder size settings, sizes read back from weights (#27); `min_temperature`. |
| 2026-10-20 | `68d2b65`, `0e9ff1f` | Window cache (#14); per-epoch checkpoints and `--resume`; pause/resume. |

**Why our encoder beat the NeuralBench baseline by 24 points** (`RESULTS.md`: "bigger head,
10× higher lr"). The baseline's configuration (`submissions/neuralbench_eegnet_things_eeg2_seed34/run_config.yaml`)
versus ours:

| | NeuralBench EEGNet (0.29–0.30) | Our encoder (0.54) |
|---|---|---|
| Network | braindecode EEGNet defaults (F1 8, D 2, F2 16, kernel 64), **77k parameters** (counted from the shipped weights) | 1.39M parameters |
| Readout | one linear layer: 48 features (16 maps × 3 steps) → 1536 | 896 features → 512 → 1536 MLP |
| Optimiser | AdamW, **lr 1e-4**, weight decay 0.05, OneCycle schedule | AdamW, **lr 1e-3**, weight decay 1e-4, constant |
| Batch | 64 | 256 (4× more negatives per step) |
| Loss | NeuralBench `ClipLoss`, fixed temperature | our `clip_loss`, τ = 0.07 fixed (learnable later) |
| Stopping | 40 epochs, patience 5 on a validation rank metric | 20 epochs |

We never ran the ablation that separates these factors, so the split between "bigger head",
"higher lr" and "bigger batch" is **not measured**.

---

## 3. The loss and the training recipe

All in `Solver.fit` and `clip_loss` in `submission/submission.py`, driven by
`scripts/train_solver.py`.

### 3.1 The contrastive (InfoNCE / CLIP) loss

For a batch of $B$ windows, let $p_i$ be the model's unit-length prediction for window $i$
and $t_i$ the DINOv2 embedding of the image that window shows (normalised inside the loss).
The similarity matrix is $s_{ij} = p_i \cdot t_j$ (cosine). With temperature $\tau$:

$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{B}\exp(s_{ij}/\tau)}
$$

Read it as a **B-way classification per row**: "which of the B images in this batch did EEG
window $i$ see?" The correct answer is the diagonal; the other $B-1$ images in the batch are
the **negatives** ("in-batch negatives"), so no extra sampling is needed. Minimising it pulls
$p_i$ towards $t_i$ and pushes it away from the other targets, which is exactly what top-k
retrieval rewards. This is **InfoNCE** (van den Oord et al., 2018), popularised by **CLIP**
(Radford et al., 2021), and used for brain decoding by Défossez et al. (2023), NICE, ATM.

Why not regression (minimise $\|p_i - t_i\|^2$)? A regression model facing noisy input
predicts the conditional mean, a blurry "average image" that sits close to everything and
ranks poorly. #23's post-hoc ridge maps confirmed it: fitted by squared error, they scored
0.046–0.052 held-out (§5).

**Options tested in round 1:**
- `symmetric=True` adds the column-wise term (for each image, which EEG window matches?),
  as in CLIP. No gain here (R3).
- `dedup=True` masks out "false negatives": other windows in the batch that show the **same**
  image (each training image has ~40 windows: 10 subjects × 4 repetitions). With ~14.5k
  training images and batch 256, a batch contains only about 2 such pairs on average
  **(our estimate)**, which explains why it made no difference (R1: 41.20% vs 41.35%).

### 3.2 The temperature

$\tau$ divides the similarities before the softmax. Small $\tau$ makes the softmax sharp, so
the loss concentrates on the hardest negatives (images that look most like the right one).
- Fixed at **0.07** at first (a common default).
- **Learnable** (`learn_temperature=True`), parameterised as CLIP does: the trainable
  parameter is $\log(1/\tau)$ (initialised to $\log(1/0.07) \approx 2.66$) and it is clamped
  at $\log 100$, i.e. **$\tau \ge 0.01$** ("logit scale at most 100"). It's trained with the
  model by the same optimiser.
- Effect: **+10 points held-out** (R2: 41.35% → 51.75%), the biggest single gain.
- In R6 the learned $\tau$ fell from 0.027 (epoch 1) to the **0.01 floor by epoch ~15** and
  stayed there. R15 lowered the floor to 0.005 (`min_temperature`): $\tau$ went to 0.0072,
  but held-out trailed R6 at every comparable epoch (best 0.564 vs 0.582 by epoch 31) when a
  restart stopped it at epoch 32. **Conclusion: no gain**, floor stays at 0.01.

### 3.3 Optimisation

| Setting | Value | Notes / source |
|---|---|---|
| Optimiser | **AdamW** | Adam with decoupled weight decay (Loshchilov & Hutter, ICLR 2019) |
| Learning rate | **1e-3** | 10× the NeuralBench baseline |
| Weight decay | **1e-4** | `weight_decay` setting (#18); applied to the temperature parameter too (negligible) |
| Batch size | **256** | `--batch-size`; more negatives per step. R10/R11 test 512 / 1024 with linearly scaled lr (Goyal et al., 2017) |
| Gradient clipping | max norm **1.0** | stabilises early training |
| Schedule | constant (default); `lr_schedule='cosine'` + `warmup_epochs` | Cosine decay from SGDR (Loshchilov & Hutter, ICLR 2017); linear warm-up from Goyal et al. 2017. Applied per step via `LambdaLR` |
| Epochs | 20 originally; **40** in R6 (+3.75 held-out) | longer training still improving at 40 |
| Dropout | 0.25 | `dropout` setting |
| Seed | `--seed` | seeds weight init, shuffling and dropout |

### 3.4 Data-side recipe

- **More data (`--use-val`, #16):** NeuralBench's own "val" split of THINGS-EEG2 is ordinary
  training data (same concepts). Adding it gives +25% windows (470,140 → 587,574 with the
  held-out concepts removed; 668,400 for final models on all concepts): **+3.5 held-out**.
- **Augmentation (R4):** random circular time shift up to ±6 samples (±50 ms) and 10% channel
  dropout. It **hurt badly** (49.85% → 36.80%). Our interpretation **(hypothesis, not
  tested)**: the flattened readout uses absolute timing (§1.3, layer 10), the circular roll
  wraps the end of the window onto the start, and 50 ms is large relative to visual response
  latencies.
- **Repetition averaging (`--avg-reps K`, #17, Gifford et al. averaged repetitions too):**
  each training item becomes the mean of K repetitions of the same (subject, image), which
  cuts independent noise by about $\sqrt{K}$. With `--avg-reps-random` K is drawn from 1..K
  per item, mixing clean and noisy examples (test inputs are single trials). There are 145,400
  (subject, image) groups, so an "epoch" is **4× shorter** than with single windows.
  **Caveat:** early stopping counts epochs, so patience 5 is effectively 4× less patient.
  R7 (random 1–4) was still rising when it stopped at 0.555 (R6: 0.590), but it got there in
  31 min, where R6 needed ~105 min for 0.59 and was at ~0.51 after 31 min. A fair test needs
  patience scaled to the epoch length.

### 3.5 Training infrastructure (no effect on results, by design)

- **Early stopping on held-out (#15):** after each epoch, `scripts/train_solver.py` scores the
  held-out split with the official objective; `fit` keeps the best epoch's weights and stops
  after `patience` epochs without improvement (5; 8 for cosine runs). Caveat for teaching:
  picking the best epoch on held-out makes the reported held-out score slightly optimistic
  (a mild selection bias); comparisons between runs remain fair because every run gets it.
- **Window cache (#14, `scripts/cache_windows.py`, `--cache`):** windows extracted once to
  memory-mapped `.npy` files (24 GB). Verified to give bit-identical windows, scores and
  training batches, with ~4× faster loading and much less memory.
- **Checkpoints and `--resume`:** `fit` saves its full state (model, optimiser, scheduler,
  temperature, best epoch, history, RNG states) after every epoch; a resumed run on CPU ends
  with bit-identical weights. `make pause-training` / `resume-training` freeze and continue a
  run (used when the laptop's GPU is needed for the screen).

---

## 4. Evaluation in detail

### 4.1 The official objective

`external/2026-competition/tracks/image_decoding/objective.py`, `Objective.evaluate_result`:

```mermaid
flowchart TB
    L["Test loader: (X, y, info) batches"] --> P["pred = model.predict(X)<br/>one 1536-d vector per window"]
    L --> Y["y = target embedding of the image shown"]
    Y --> G["Gallery = np.unique(y): the distinct images<br/>(200 for THINGS-EEG2 test)"]
    P --> S1["Pooled: cosine(pred, gallery) per window"]
    P --> GM["group_means over (subject, image):<br/>average a subject's predictions for the same image"]
    GM --> S2["Subject-aggregated: cosine(mean pred, gallery)<br/>2,000 queries = 10 subjects × 200 images"]
    S1 --> K1["top5_acc, top1_acc"]
    S2 --> K2["top5_acc_subject_agg, top1_acc_subject_agg"]
    G --> S1
    G --> S2
```

- **Gallery:** the unique target embeddings of the split, found with `np.unique(y_true, axis=0)`.
  Repeated presentations of an image share an identical embedding, so they collapse to one
  candidate.
- **Scores:** predictions and candidates are L2-normalised; the score is the dot product
  (cosine similarity).
- **Top-k accuracy** (`benchmark_utils/metrics.py`, `topk_accuracy`): a query is a hit if its
  true image's rank is below k. Ties share their average rank. **Chance level**: 5/200 =
  **2.5%** for top-5, 0.5% for top-1.
- **Subject-aggregated** (`group_means`): average the predictions of all windows with the same
  (subject, image) key, then rank those means. On the test set each subject saw each image
  ~80 times, so this averages ~80 predictions per query: it's why our top-5 is 0.74
  subject-aggregated but 0.31 pooled. The sealed phase ranks on top-5 subject-aggregated;
  the warm-up leaderboard's main column has matched our subject-aggregated top-5 on every upload.
  (The objective's docstring says warm-up uses the pooled metric; the leaderboard columns show
  both, and README/RESULTS track subject-aggregated. **Worth confirming with the organisers.**)

### 4.2 Our held-out-concept split

`src/data/heldout.py` + `configs/heldout_concepts.json` (seed 0, committed):
- 200 of the 1,654 training **concepts** are removed from training entirely (no window of
  those concepts is trained on), so the model must generalise to unseen concepts, like the
  test set.
- One image per held-out concept (e.g. `air_conditioner_06s`) forms a **200-image gallery**,
  the same size as the test gallery; its windows (from NeuralBench's train and val splits)
  are the queries: 8,076 windows ≈ 200 images × 10 subjects × ~4 repetitions.
- Training then uses the remaining 1,454 concepts (587,574 windows with `--use-val`).

**Why the held-out score is lower than the test score for the same model:** the held-out
images were shown only **~4 times** per subject (training-set protocol) while test images
were shown **~80 times**. Subject aggregation averages ~4 predictions instead of ~80, so much
less noise is removed. The ranking of models is what matters, and it has been consistent.

### 4.3 From held-out to the leaderboard (empirical)

| Run | Held-out top-5 SA | Test top-5 SA (same model) | Gap |
|---|---|---|---|
| control (seed 1) | 0.4135 | 0.5450 | +13.2 |
| R2 learnable τ | 0.5175 | 0.6525 | +13.5 |
| R5 + val data | 0.5525 | 0.7110 | +15.9 |

Then retraining the chosen recipe on **all** concepts with 3 seeds and ensembling added
+2.7 on test (R5 0.7110 → 0.7380). So a held-out gain of x points has translated into
roughly x points on the leaderboard. R6's 0.590 suggests about **0.75–0.78** for its final
ensemble **(a forecast, not a measurement)**.

**Why local test = Codabench:** same data pipeline (NeuralBench pinned to the worker's
version, `neuralbench==0.4.0` etc.), same test split, same objective code, deterministic
inference. All six uploads so far matched local scores to two decimals.

---

## 5. Every experiment and its result

Held-out = top-5 subject-aggregated on our held-out split (seed 1 unless noted). Test =
top-5 subject-aggregated on THINGS-EEG2 test, computed locally. Codabench = leaderboard score.
Sources: `RESULTS.md`, `logs/overnight_summary.md`, `logs/round2.status`, `logs/*.log`,
`submissions/*/run_config.json`.

| Run | Hypothesis | Change | Held-out | Test | Codabench | Conclusion |
|---|---|---|---|---|---|---|
| NeuralBench EEGNet, seed 34 | Official baseline as a floor | braindecode EEGNet, NeuralBench recipe | – | 0.288 | **0.29** | Starting point (submission 1001528) |
| EEGNet ensemble 34+35 | Seed averaging helps | mean of 2 models' normalised predictions | – | 0.2975 | 0.30 | +1; 3 seeds tied (0.2970) |
| REVE frozen + ridge | A foundation model's features suffice | official ridge probe on frozen REVE | – | 0.025 | – | Chance level |
| REVE frozen + linear CLIP head | Same, with a contrastive head | linear head, CLIP loss | – | 0.127 (120 Hz) / 0.086 (200 Hz) | – | Far below EEGNet; deprioritised |
| Our encoder (unseeded) | Bigger model + higher lr + CLIP loss | §2.2 table, τ 0.07, 20 epochs | 0.4215 | 0.5415 | **0.54** | +24 over baseline |
| 3-encoder ensemble | Seeds help | 3 seeds | 0.4315 | 0.5530 | 0.55 | +1 |
| Control (seed 1) | Reference for round 1 | same, seed 1 | 0.4135 | 0.5450 | – | |
| R1 dedup | False negatives hurt | `dedup=True` | 0.4120 | – | – | No gain (few duplicates per batch) |
| **R2 learnable τ** | τ = 0.07 is too soft | `learn_temperature=True` | **0.5175** | 0.6525 | **0.65** | **+10.4**: keep |
| R3 τ + dedup + symmetric | CLIP's symmetric loss helps | + `dedup`, `symmetric` | 0.4985 | – | – | Worse than R2: drop |
| R4 R3 + augmentation | Invariance to shifts/channels | ±50 ms circular shift, 10% channel dropout | 0.3680 | – | – | Hurts a lot: drop |
| **R5 R2 + val data** | More data helps | `--use-val` (+25%) | **0.5525** | 0.7110 | **0.71** | **+3.5**: keep |
| #23 recalibration | Post-hoc linear map fixes predictions | centering / ridge maps on R2's outputs | 0.376 / 0.046–0.052 (none: 0.5175) | – | – | All hurt: closed |
| **Final 3-seed ensemble** | All concepts + ensemble | R5 recipe on all 668,400 windows, seeds 1–3 | n/a | **0.7380** (single seeds 0.715–0.721) | **0.74** (top-1 0.40) | Current best (submission 1004941) |
| **R6 40 epochs** | 20 epochs were too few | `n_epochs=40`, early stopping | **0.590** (epoch 36) | not run | – | **+3.75**: keep; still rising at 40 |
| R15 floor 0.005 | τ is stuck at the 0.01 floor | `min_temperature=0.005` | 0.564 at epoch 31 (stopped at 32/40 by a restart) | – | – | Trailed R6 throughout: no gain |
| R7 avg 1–4 reps (random) | Cleaner training examples | `--avg-reps 4 --avg-reps-random` | 0.555 (epoch 46, stopped at 51) | – | – | Cut short by patience (§3.4); faster per minute than R6; re-test with scaled patience |
| R8 avg 2 reps | Same, fixed k = 2 | `--avg-reps 2` | running at the time of writing (0.508 at epoch 28) | – | – | – |

Top-1 subject-aggregated for the uploaded models: 0.09 (EEGNet) → 0.22 → 0.34 (R2) → 0.37
(R5) → **0.40** (final ensemble).

Queued (`logs/round2.status`): R9–R11 cosine schedule with batch 256 / 512 / 1024 (#18);
R12–R14 wider convolutions / head 1024 + residual block / both (#27).

---

## 6. Submission and inference path

```mermaid
flowchart LR
    T["train_solver.py<br/>load_model (fresh init) → fit → save_model"] --> F["submissions/NAME/<br/>submission.py + weights.pt<br/>(+ run_config*.json)"]
    F --> Z["make zip-submission SUBMISSION=submissions/NAME<br/>(run_config* left out)"]
    Z --> CB["Codabench: unzip read-only,<br/>Solver.load_model(meta), then predict(X) per batch"]
    CB --> SC["objective.evaluate_result → leaderboard"]
```

- **Contract** (`benchmark_utils.base_solver.CompetSolver`): Codabench never trains. It calls
  `Solver.load_model(meta)` with `n_chans`, `n_times`, `n_outputs`, `ch_names`, `chs_info`,
  `device`, `submission_dir`, and then `model.predict(X)` on `(B, C, T)` batches; `predict`
  must return `(B, n_outputs)`.
- **Inference vs training in `load_model`:** if `train_loader` is set (a local training
  run) it builds a fresh, seeded model; otherwise it loads the shipped weights.
- **Sizes from the weights (`encoder_sizes`, #27):** the number of filters, the head width and
  the number of residual blocks are read from the tensor shapes in the weights file, so the
  uploaded `submission.py` never needs editing after a wider run.
- **Ensembles (`EmbeddingEnsemble`):** if the folder contains `weights_*.pt` files, one model
  is built per file and their (unit-length) predictions are averaged.
- **Run time on Codabench:** 226–318 s per upload so far (limit 60 min), e.g. 305 s for the
  3-model ensemble.
- **Reproducibility:** each run writes `run_config.json` (command, commit, settings, seed,
  windows, minutes, best epoch) and `run_config_curve.json` (per-epoch loss, temperature, lr,
  held-out), which the top-3 audit needs.

---

## 7. Paper map and bibliography

### 7.1 Idea → paper → where in our code

| Idea in our code | Paper | What to read | Where it lives |
|---|---|---|---|
| Temporal-then-spatial conv factorisation | Schirrmeister et al. 2017 | ShallowConvNet / DeepConvNet architecture figures and their FBCSP motivation | `EEGToImageEncoder.temporal_conv`, `.spatial_conv` |
| Compact EEG CNN: block order, ELU, pooling, dropout, separable conv | Lawhern et al. 2018 (EEGNet) | Architecture section and table of layers | `EEGToImageEncoder` (whole conv part) |
| Depthwise-separable convolution | Chollet 2017 (Xception) | Section on depthwise-separable convs | `EEGToImageEncoder.sep_conv` |
| BatchNorm | Ioffe & Szegedy 2015 | | every `BatchNorm*` |
| ELU / GELU activations | Clevert et al. 2016 / Hendrycks & Gimpel 2016 | | `spatial_conv`, `sep_conv` / `projection_head` |
| Dropout | Srivastava et al. 2014 | | `Dropout(dropout)` |
| Contrastive loss with in-batch negatives | van den Oord et al. 2018 (InfoNCE); Radford et al. 2021 (CLIP) | CLIP §2.3 and its Figure 3 pseudo-code | `clip_loss` |
| Learnable temperature, logit scale ≤ 100 | Radford et al. 2021 (CLIP) | CLIP §2.5 (training details) | `Solver.fit` (`log_scale`, `min_temperature`) |
| Symmetric loss | Radford et al. 2021 | Figure 3 | `clip_loss(symmetric=True)` |
| CLIP loss for brain signals, retrieval evaluation | Défossez et al. 2023; Benchetrit et al. 2024 | Methods (loss, retrieval metric) | `clip_loss`, objective |
| Zero-shot EEG→image retrieval on THINGS-EEG2 | Song et al. 2024 (NICE); Li et al. 2024 (ATM) | Problem setup and results tables | whole project |
| Residual projector + LayerNorm | Li et al. 2024 (ATM); NICE's projector | Architecture section | `ResidualBlock`, `head_blocks` |
| DINOv2-giant target embeddings | Oquab et al. 2023 (DINOv2) | Intro + feature quality sections | targets (NeuralBench, `facebook/dinov2-giant`) |
| AdamW | Loshchilov & Hutter 2019 | | `Solver.fit` optimiser |
| Cosine schedule | Loshchilov & Hutter 2017 (SGDR) | | `Solver._lr_factor` |
| Warm-up, lr scaling with batch size | Goyal et al. 2017 | §2.1–2.2 | `_lr_factor`, R10/R11 |
| Repetition averaging / data protocol | Gifford et al. 2022 (THINGS-EEG2) | Paradigm and preprocessing | `src/data/repetitions.py` |
| Concepts and images | Hebart et al. 2019 (THINGS) | | `configs/heldout_concepts.json` |
| Blurred-image targets (issue #26) | UBP (CVPR 2025) | | not implemented yet |
| Position-based spatial attention, spatial dropout | Défossez et al. 2023 | "Spatial attention" layer in Methods (Fourier embedding of sensor positions) | `SpatialAttention`, `Solver.virtual_channels` (§9.2) |
| Electrode positions (10-05 system) | Oostenveld & Praamstra 2001 **(unverified)**; MNE's `standard_1005` montage | | `ELECTRODE_XY`, `channel_positions` |
| Held-out participants (cross-subject validation) | standard practice; motivated by the sealed final's unseen subjects | | `heldout_subjects_split` (§9.1) |
| EEG foundation model (tried, weak) | El Ouahidi et al. 2025 (REVE) | | `logs/reve*` |
| Final-phase data family | Xu et al. 2024 (Alljoined1); Xu et al. 2025 (Alljoined-1.6M) | Dataset papers | issues #4, #21 |
| A convolution + transformer EEG encoder (candidate for #12) | Song et al. 2023 (EEG Conformer) | Architecture figure | not implemented |

### 7.2 Bibliography

Verification: "verified" means the citation was checked against metadata shipped in our
installed packages (braindecode, NeuralFetch, PyTorch, transformers). Web search was not
available while writing this, so the others are from memory and marked **(unverified)**.

**Datasets**
- Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset
  for modeling human visual object recognition. *NeuroImage*. doi:10.1016/j.neuroimage.2022.119754
  — verified (NeuralFetch `gifford2022large`). Data: osf.io/3jk45.
- Hebart, M. N., et al. (2019). THINGS: A database of 1,854 object concepts and more than
  26,000 naturalistic object images. *PLoS ONE* 14(10): e0223792. **(unverified)**
- Xu, J., Aristimunha, B., Feucht, M. E., Qian, E., et al. (2024). Alljoined — A dataset for
  EEG-to-Image decoding. arXiv:2404.05553 — verified (NeuralFetch `xu2024alljoined`; 8 subjects).
- Xu, J., Nunes, U. B., Jiang, W., Ryther, S., Pringle, J., Scotti, P. S., Delorme, A., &
  Kneeland, R. (2025). Alljoined-1.6M: A million-trial EEG-image dataset for evaluating
  affordable brain-computer interfaces. arXiv:2508.18571 — verified (NeuralFetch
  `xu2025alljoined`; 20 subjects, Emotiv headset).

**EEG architectures**
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J.
  (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer
  interfaces. *Journal of Neural Engineering* 15(5), 056013. arXiv:1611.08024 — verified (braindecode).
- Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M.,
  Eggensperger, K., Tangermann, M., Hutter, F., & Ball, T. (2017). Deep learning with
  convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*.
  doi:10.1002/hbm.23730 — verified (braindecode).
- Song, Y., Zheng, Q., Liu, B., & Gao, X. (2023). EEG Conformer: Convolutional transformer for
  EEG decoding and visualization. *IEEE Transactions on Neural Systems and Rehabilitation
  Engineering* 31, 710–719 — verified (braindecode; braindecode dates it 2022, the journal
  volume is 2023).
- Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. *CVPR* —
  verified (cited in braindecode's EEGNet).
- El Ouahidi, Y., Lys, J., Thölke, P., Farrugia, N., Pasdeloup, B., Gripon, V., Jerbi, K., &
  Lioi, G. (2025). REVE: A foundation model for EEG — adapting to any setup with large-scale
  pretraining on 25,000 subjects. *NeurIPS 2025*. openreview.net/forum?id=ZeFMtRBy4Z —
  verified (braindecode).

**EEG/MEG → image decoding**
- Song, Y., Liu, B., Li, X., Shi, N., Wang, Y., & Gao, X. (2024). Decoding natural images from
  EEG for object recognition (NICE). *ICLR 2024*. arXiv:2308.13234 **(unverified)**
- Li, D., Wei, C., Li, S., Zou, J., & Liu, Q. (2024). Visual decoding and reconstruction via EEG
  embeddings with guided diffusion (ATM). *NeurIPS 2024*. arXiv:2403.07721 **(unverified)**
- "Bridging the vision-brain gap with an uncertainty-aware blur prior" (UBP). *CVPR 2025*.
  **(unverified: authors and arXiv ID; search the title)**
- Benchetrit, Y., Banville, H., & King, J.-R. (2024). Brain decoding: toward real-time
  reconstruction of visual perception. *ICLR 2024*. arXiv:2310.19812 **(unverified)**
- Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J.-R. (2023). Decoding speech
  perception from non-invasive brain recordings. *Nature Machine Intelligence* 5(10),
  1097–1107 — verified (braindecode REVE references). arXiv:2208.12266 **(ID unverified)**

**Representation learning and optimisation**
- Radford, A., et al. (2021). Learning transferable visual models from natural language
  supervision (CLIP). *ICML 2021*. arXiv:2103.00020 **(unverified)**
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive
  predictive coding. arXiv:1807.03748 **(unverified)**
- Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision.
  arXiv:2304.07193 — verified (transformers' DINOv2 module).
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization (AdamW). *ICLR
  2019*. arXiv:1711.05101 — verified (PyTorch `AdamW` docstring).
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts.
  *ICLR 2017*. arXiv:1608.03983 — verified (PyTorch `lr_scheduler`).
- Goyal, P., et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour.
  arXiv:1706.02677 **(unverified)**
- Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML 2015*. **(unverified)**
- Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2016). Fast and accurate deep network
  learning by exponential linear units (ELUs). *ICLR 2016*. **(unverified)**
- Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv. **(unverified)**
- Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from
  overfitting. *JMLR*. **(unverified)**

---

## 8. Teaching kit

### 8.1 A 60–90 minute session

| Time | Topic | Show |
|---|---|---|
| 0–10 min | The task and why it's retrieval, not classification; chance level 2.5% | how_it_works.md, first diagram |
| 10–20 | The data: THINGS-EEG2, windows, why repetitions matter (4 vs 80) | how_it_works.md §2; §4.2 here |
| 20–35 | The encoder, layer by layer: temporal → spatial → separable → head; receptive field | §1.1 diagram and §1.2 table |
| 35–50 | The loss: the B×B matrix, negatives, temperature; why not regression | §3.1 formula; how_it_works.md loss diagram |
| 50–60 | Evaluation and the held-out split; why held-out < test | §4.1 diagram, §4.3 table |
| 60–75 | The experiment log: what worked (τ, data, epochs), what didn't, and why | §5 table |
| 75–90 | What's next: final phase = 32 channels + unseen people; headset-independent spatial layer; held-out participants | how_it_works.md §8 |

Tip: before the session, run exercise 2 so the scores appear live on screen.

### 8.2 Check-your-understanding questions (with answers)

1. **Why can the model recognise images of concepts it never trained on?**
   It predicts a point in DINOv2's embedding space, not a class. DINOv2 places any image,
   including unseen ones, in that space, so retrieval works for new images (zero-shot).
2. **What is chance-level top-5 accuracy on the test set?** 5/200 = 2.5%.
3. **Why is subject-aggregated accuracy (0.74) so much higher than pooled (0.31)?**
   It averages ~80 predictions per (subject, image) before ranking, which cancels most of
   the single-trial noise.
4. **What does the temporal convolution learn, in signal-processing terms?**
   32 FIR filters of 258 ms shared across electrodes: band-pass filters or response-shape
   detectors.
5. **Why does the model have to be retrained for the 32-channel final phase?**
   The spatial conv's kernel spans exactly 63 electrodes in a fixed order; its weights have no
   meaning for a different montage.
6. **Where are 90% of the parameters?** In the projection head (896 → 512 → 1536).
7. **Write the InfoNCE loss for one row and say what the negatives are.**
   $-\log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$; the negatives are the other images in
   the same batch.
8. **What does lowering the temperature do?** Sharpens the softmax so the loss focuses on the
   hardest wrong images. Learning it gave +10 held-out; going below 0.01 gave nothing.
9. **Why did `dedup` make no difference?** Same-image duplicates are rare in a batch of 256
   (about 2 pairs), so there were few false negatives to remove.
10. **Give one plausible reason augmentation by time shift hurt.**
    The flattened readout relies on absolute timing of responses; shifting by up to ±50 ms
    (with wrap-around) corrupts that.
11. **Why is the held-out score lower than the test score for the same model?**
    Held-out images were seen ~4 times per subject, test images ~80, so subject aggregation
    removes less noise on held-out.
12. **Why don't we choose models by their test score?** The test set is the public
    leaderboard data; selecting on it overfits to it and tells us nothing about the sealed
    phase. We select on the held-out split.
13. **Why do we retrain on all concepts for uploads?** More concepts cover DINOv2's space better;
    it added ~+2.7 on test together with the 3-seed ensemble.
14. **Why was R7 (repetition averaging) not a fair loss?** Its epochs are 4× shorter, so
    patience 5 stopped it while it was still improving.
15. **What does `encoder_sizes` solve?** It reads the layer sizes from the weights, so the same
    `submission.py` can load a model of any width without edits.

### 8.3 Hands-on exercises

**Exercise 1: count the parameters and check the shapes (CPU, 1 minute).**
```python
# save as count_params.py at the repo root, run: .venv/bin/python count_params.py
import importlib.util, sys, torch
sys.path.insert(0, "external/2026-competition/tracks/image_decoding")
spec = importlib.util.spec_from_file_location("s", "submission/submission.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
enc = m.EEGToImageEncoder(n_chans=63, n_times=120, output_dim=1536).eval()
x = torch.zeros(2, 1, 63, 120)
for block in (enc.temporal_conv, enc.spatial_conv, enc.sep_conv):
    for layer in block:
        y = layer(x)
        print(f"{str(layer)[:60]:60s} {tuple(x.shape)} -> {tuple(y.shape)} "
              f"{sum(p.numel() for p in layer.parameters())}")
        x = y
print("total:", sum(p.numel() for p in enc.parameters()))  # 1,387,872
```
Questions: where do 129,024 parameters come from? (32 × 63 × 64.) What changes with
`EEGToImageEncoder(32, 120, 1536)`, the final phase's channel count?

**Exercise 2: score an existing model like Codabench does (≈1 min with the cache).**
```bash
python scripts/eval_local.py submissions/e6_r5_learntemp_useval --cache cache/windows/things_eeg2
# expect 0.7110 / 0.3745 / 0.2824 / 0.0958 (R5, Codabench 0.71)
python scripts/eval_local.py submissions/e6_r5_learntemp_useval --split heldout --cache cache/windows/things_eeg2
# expect 0.5525 held-out top-5 subject-aggregated
```
Questions: why is the second number lower? Which column is the leaderboard metric?

**Exercise 3: plot learning curves (CPU, 2 minutes).**
```python
# save as plot_curves.py, run: .venv/bin/python plot_curves.py
import json, matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
for run in ["m2_r6_epochs40", "m2_r7_avgreps4rand"]:
    curve = json.load(open(f"submissions/{run}/run_config_curve.json"))
    ep = [r["epoch"] for r in curve]
    a.plot(ep, [r["heldout"] for r in curve], label=run)
    b.plot(ep, [r["temperature"] for r in curve], label=run)
a.set(xlabel="epoch", ylabel="held-out top-5 (subject-agg)"); b.set(xlabel="epoch", ylabel="temperature", yscale="log")
a.legend(); plt.tight_layout(); plt.savefig("curves.png"); print("wrote curves.png")
```
Questions: when does the temperature reach its 0.01 floor? Re-plot with the x-axis in
training samples (epochs × 587,574 for R6, × 145,400 for R7): which run learns faster per
sample?

---

## 9. Next: new people and new headsets (in progress)

Everything above was tuned on THINGS-EEG2, where the test participants are the **same 10
people** the model trained on, recorded with **63 channels**. The sealed final is different:
**11 new people, a 32-channel Emotiv headset, new images.** Two pieces of code (commits
`25b12fb` and `d3ed4ea`, Oct 20) prepare for that. Neither has been trained at scale yet.

### 9.1 Measuring what the final measures: held-out participants

Our held-out split (§4.2) holds out *concepts* but not *people*, so it can't tell us how well a
model works on someone it has never seen, which is notoriously the hard part of EEG (skull
thickness, cortical folding and cap placement differ between people).

`train_solver.py --holdout-subjects` / `eval_local.py --split heldout_subjects`
(`heldout_subjects_split` in `src/data/heldout.py`):

- the 200 held-out concepts are removed from training, as before;
- **participants 6 and 7** (`configs/heldout_subjects.json`, random with seed 0) are removed
  entirely: training uses 8 people;
- the score is computed on those 2 people's windows of the held-out concepts. Each held-out
  concept has 10 images, so there are **10 galleries of 200 images** (gallery *g* = the *g*-th
  image of every concept), each scored with the official metric, then averaged.

Why 10 galleries: one gallery with 2 people gives only 400 subject-aggregated queries of ~4
repetitions each (standard error ≈ 2.5 points, too noisy to judge 1–2-point changes). Ten
galleries give 16,174 windows, ~4,000 queries, and a measured standard error of **≈ 0.8 points**
(`eval_local` prints it as `top5_acc_subject_agg_se`).

| Model | Trained on participants 6 and 7? | Held-out-participants score (top-5 SA) |
|---|---|---|
| R6 | yes (reference, not a real held-out score) | 0.6352 ± 0.0077 |
| any `--holdout-subjects` run | no | to be measured: the drop from 0.635 is the cost of unseen people |

### 9.2 One model for any headset: spatial attention

Our `spatial_conv` (§1.3) is a convolution with kernel height = number of channels, in a fixed
order. Its weights only make sense for THINGS-EEG2's 63 channels, so a THINGS-EEG2 model can't
read Emotiv's 32 channels, and the two datasets can't be trained together without remapping.

Défossez et al. (2023) solved this for MEG with a **spatial attention** layer, which we adapt
(`SpatialAttention`, enabled with `--set virtual_channels=64`):

```mermaid
flowchart LR
    N["Channel names<br/>(meta['ch_names'])"] --> P["2D scalp position of each electrode<br/>(standard 10-05 table, ELECTRODE_XY)"]
    P --> F["Fourier features of the position<br/>cos/sin(2π(kx·x + ky·y)), 16×16 frequencies<br/>→ 512 numbers per electrode"]
    F --> S["Score of electrode c for virtual channel j<br/>= learned weights_j · features_c"]
    S --> SM["softmax over electrodes<br/>(unknown names masked out;<br/>training: drop electrodes near a random point)"]
    X["EEG (B, C, 120)<br/>C = 63 or 32"] --> M["Each virtual channel = weighted<br/>average of the electrodes"]
    SM --> M
    M --> V["(B, 64, 120): same shape for any headset"] --> E["the existing encoder<br/>(temporal conv → spatial conv → …)"]
```

- **Why it's montage-independent:** the learned parameters (`heads`, 64 × 512 ≈ 33k numbers)
  define a smooth *function of scalp position*, not one weight per channel. A new headset just
  evaluates that function at its own electrode positions.
- **Positions:** the pipeline only passes electrode *names*, and Codabench's worker only has
  `torch`, so `submission.py` embeds a table of 343 standard 10-05 positions, generated once
  from MNE's `standard_1005` montage (sphere-fitted, then flattened by an azimuthal-equidistant
  projection from the top of the head, like a topographic map), at a fixed scale, so 63- and
  32-channel headsets share one coordinate frame.
- **Spatial dropout** (`spatial_dropout=0.15`): during training, every electrode within that
  radius of a random scalp point is ignored, so the model learns not to depend on any one
  region, which is useful when the final's headset lacks some of THINGS-EEG2's electrodes.
- **Checked** (synthetic data): with `virtual_channels=0` (the default) training is bit-identical
  to before; a model trained on 63 channels reloads to identical predictions and runs on a
  32-channel Emotiv montage with the same weights. Positions are per-montage buffers, not saved
  in `weights.pt`.

### 9.3 The plan

1. Train the R6 recipe with `--holdout-subjects` (the unseen-people baseline).
2. Same with `--set virtual_channels=64 spatial_dropout=0.15`: does spatial attention cost
   anything on THINGS-EEG2, and does it help on unseen people?
3. Then joint training on THINGS-EEG2 + Alljoined (issue #21), possible only with step 2's layer,
   validated on Alljoined's held-out participants (issue #4).
