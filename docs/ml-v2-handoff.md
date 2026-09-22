# v2 deepfake detector — handoff

Self-contained. Written 2026-09-22 after three working sessions. Supersedes
`ml-v2-resume.md` (deleted; everything in it is carried over here).

Nothing is running. Nothing is committed. All state is on disk under
`data/`, `machine-learning/checkpoints/`, `machine-learning/runs/` and
`logs/experiments/`.

---

## 1. The problem

The shipped v1 model (`machine-learning/checkpoints/efficientnet_b0_20260901_204509`)
is an EfficientNet-B0 fine-tuned end to end on **one** manipulation family,
FaceForensics++ Deepfakes c23. Measured at the start of this work:

| Test set | v1 frame AUC |
|---|---|
| FF++ Deepfakes c23 (its own test split) | 0.9989 |
| Celeb-DF-v2 (unseen corpus) | 0.8400 |
| Celeb-DF-v2 @ JPEG q40 | 0.7581 |
| Celeb-DF-v2 @ 0.5x downscale | 0.7838 |

It also returns 1.0 on a uniform-noise image. The 0.999 is the training
manipulation memorised, not detection ability.

## 2. What v2 changes

Three levers, and the results in §4 show the first two both paid:

1. **Corpus diversity** — 13 manipulation families instead of 1, spanning
   graphics face-swap, learned face-swap, reenactment/talking-head, GAN
   synthesis and diffusion synthesis.
2. **Frozen foundation backbone** — CLIP ViT with only LayerNorm affines and
   the head trained (41k-103k parameters), after LNCLIP-DF (arXiv:2503.19683).
   Full fine-tuning is what let v1 overwrite generic features with
   method-specific ones.
3. **Degradation augmentation + Self-Blended Images** — training crops are
   pristine PNG, deployment is codec output. SBI (Shiohara & Yamasaki, CVPR
   2022) manufactures pseudo-fakes from genuine faces alone, teaching the
   blending boundary every face swap shares regardless of generator.

## 3. Data on disk

`data/dl` 89 GB of source zips (35 files, all downloaded, nothing missing).
`data/processed_v2` 26 GB, **305,666 rows** in `manifest.csv` — up from 288,072
after session 4 ingested four zips an earlier build had silently dropped and
removed six rows pointing at zero-length files (§6b).

| Source | HF repo | Role |
|---|---|---|
| Celeb-DF-v2, FF++, DFDCP, UADFV | `Ella4839/DeepfakeBench` | real frames + classic/cross-corpus fakes |
| DF40 train methods (14 zips) | `ManhQuangAI/DF40_train` | held-in modern manipulations |
| DF40 test methods (17 zips) | `ManhQuangAI/df-40-test-full` | held-out modern manipulations |

`manifest.csv` columns: `path,label,dataset,method,identity,split,group,landmark`.

```
dataset  group      label            dataset  group      label
cdf      heldin     real     14191   ffpp     heldin     fake     15978
         heldout    fake     22556                       real     10884
df40     heldin     fake     61812            heldout    fake     40303
         heldout    fake     61333            valunseen  fake      8000
                    real     13328   dfdcp    heldout    fake     31481
         valunseen  fake     16900                       real      8116
                                     uadfv    heldout    fake/real   392/392
```

### Crop uniformity — the claim, and where it does not hold

The video corpora are uniform: every crop from Celeb-DF-v2, FF++, DFDCP, UADFV
and the DF40 *video* manipulations is a 256x256 DeepfakeBench-aligned face
crop. That uniformity is load-bearing, because cropping the reals with a
different detector would hand the model a trivial shortcut ("MTCNN framing =
real, DeepfakeBench framing = fake").

DF40's **entire-face synthesis** sets are not uniform, and nothing had noticed:

| set | crop size |
|---|---|
| MidJourney, whichfaceisreal, styleclip (fakes) | 1024x1024 |
| heygen_new (reals) | 512x512 |
| stargan, starganv2, MidJourney (reals) | 256x256 |
| CollabDiff, styleclip (reals) | 218x178 (CelebA) |

Everything is resized to 224 before the model sees it, so a 1024px image is
downsampled 4.6x and a 256px one barely at all — which is exactly the operation
that destroys generator high-frequency artifacts. Since whichfaceisreal's
1024px reals are in the test split, the evaluation real pool is itself mixed.

Measured, on the ViT-L model, the confound is real but small and does not
rescue anything: 1024px reals score a median 0.175 against 0.079 for the rest,
so scoring MidJourney against 1024px reals alone gives AUC 0.135 and against
256px reals alone 0.350 — worse and better than the pooled 0.318, but a failure
either way. The video manipulations, including SadTalker, are unaffected: they
and their reals are all 256px.

### Method partition (`machine-learning/protocol.py`)

* **held-in** (trainable): Deepfakes, Face2Face, faceswap, blendface,
  facedancer, fsgan, fomm, facevid2vid, lia, wav2lip, StyleGAN2, ddim, DiT
* **val-unseen** (model selection only): FaceSwap, mobileswap, tpsm, VQGAN
* **held-out** (the honest number): inswap, simswap, uniface, e4s, deepfacelab,
  sadtalker, heygen, hyperreenact, mcnet, NeuralTextures, FaceShifter,
  DeepFakeDetection, SiT, StyleGAN3, MidJourney, CollabDiff, stargan,
  starganv2, styleclip, whichfaceisreal, plus the Celeb-DF-v2, DFDCP and
  UADFV corpora

InSwapper and SimSwap are held out deliberately: they are the swaps someone is
most likely to actually run today, so they are the question, not the answer.

Split is by hashed **target** identity, consistent across every corpus, so a
person never straddles splits. A swap's *source* identity can appear in another
split — the FF++/DeepfakeBench convention. The cross-corpus held-out sets
(DFDCP, UADFV) have no identity overlap at all, so headline numbers are clean.

## 4. Results

Selection is on val-unseen macro AUC. The **held-out** column is the honest
number: mean per-method AUC over the manipulations and corpora excluded from
training, each scored against the shared real pool. The table below covers 14
of them (35 manipulations are in the test split in total, but 21 are held-in or
val-unseen); the re-scored table produced by `finalize_v2.sh` covers 23, since
§6b added four methods and §6c made five more visible.

| run | trainable | val-unseen | **held-out** | all-method | video | real FPR@0.5 |
|---|---|---|---|---|---|---|
| `effnet_data_only` | 4.17 M | 0.9048 | **0.6715** | 0.7985 | 0.7874 | 0.220 |
| `clipb_full` | 0.041 M | 0.9679 | **0.8200** | 0.8934 | 0.8871 | 0.086 |
| `clipl_full` | 0.103 M | 0.9803 | **0.8611** | 0.9204 | 0.9247 | 0.079 |

`effnet_data_only` is the control: v1's architecture on v2's corpus. Held-out
AUC **0.67 -> 0.82** came from swapping it for a frozen CLIP ViT-B/16 training
41k LayerNorm parameters — same corpus, same augmentation, same SBI, same
sampler, 100x fewer trained parameters. Real FPR fell to 0.086 with no
threshold tuning, because a better-separated score distribution puts the
balanced operating point somewhere sensible by itself.

**Neither CLIP run overfits.** Unseen AUC sits *above* in-domain val at every
epoch (ViT-B 0.9679 vs 0.9500; ViT-L 0.9803 vs 0.9729), while the EfficientNet
run crossed over at epoch 4 — in-domain still rising as unseen fell. LN-tuning
lacks the capacity to memorise the training manipulations, which is exactly the
mechanism the approach was betting on. It also means **selecting on in-domain
val would have shipped a worse detector**; the val-unseen protocol earns its
keep.

Per-epoch val-unseen:

```
effnet  0.8734 0.8972 0.9027 0.9021 0.9000   best 3, then decays
clipb   0.9522 0.9609 0.9669 0.9664 0.9679   best 5, still climbing
clipl   0.9688 0.9741 0.9799 0.9803 0.9795   best 4, converged
```

ViT-L passed ViT-B's best-of-5 after one epoch. 5 epochs is right for ViT-L;
ViT-B is probably slightly undertrained.

### Corruption robustness

| condition | `effnet` | `clipb` | `clipb` FPR@0.5 | `clipl` | `clipl` FPR@0.5 |
|---|---|---|---|---|---|
| clean | 0.6715 | 0.8200 | 0.086 | **0.8611** | 0.079 |
| JPEG q40 | 0.6512 | 0.7932 | 0.143 | **0.8245** | 0.075 |
| downscale 0.25x | 0.6286 | 0.7609 | 0.189 | **0.8013** | 0.151 |
| JPEG q10 | 0.6293 | 0.6910 | **0.493** | **0.7259** | 0.186 |

ViT-L wins every condition, and the q10 calibration collapse is far milder:
false-positive rate 0.186 against ViT-B's 0.493. Scale helps the *operating
point* more than it helps the ranking — held-out AUC gains 4 points at q10
while the false-positive rate falls by a factor of 2.7. That is still a
collapse relative to its own 0.079 clean, which is what §5b's `heavy_wide`
retrain is for.

## 5. The two open failure modes

### 5a. SadTalker — resolved: no per-frame signal, and not a class failure

Diagnosed with `machine-learning/diagnose_v2.py`, which reports where a method's
scores *sit* rather than only how well they rank. Three questions, three
answers.

**Is it inverted, or is there simply no signal?** Close to the latter, and it
matters because the two call for opposite fixes. On the ViT-L model SadTalker
scores 0.433 — seven points below chance, with a median of 0.063 against a
real-pool median of 0.091 (−0.17 real-pool IQRs) and a histogram the same shape
as the reals'. That is a detector with essentially nothing to go on, leaning
very slightly toward "real".

Contrast the one method in the corpus that *is* properly inverted: **MidJourney
at 0.318**, median 0.047 — half the real pool's — so a genuine face is ranked
above a MidJourney generation 68% of the time. That is a cue firing backwards.
SadTalker is not that; it is a blank.

(MidJourney was invisible until this session — see §6c. It is a larger miss
than SadTalker and nothing in the project had ever scored it.)

**Is a whole class of talking-head generation invisible?** No. The
`reenactment` family averages 0.85 and contains held-out members the detector
handles well:

| held-out reenactment method | ViT-L AUC |
|---|---|
| hyperreenact | 0.9587 |
| mcnet | 0.8272 |
| NeuralTextures | 0.8181 |
| **heygen_new** | **0.5529** |
| **sadtalker** | **0.4161** |

So the corpus's five held-in reenactment methods do transfer — just not to
these two. The handoff's own decision rule ("if it is a class failure, add a
held-in talking-head method") therefore does **not** fire.

**Is it a capacity problem?** No, and this is the clearest signal in the whole
project. Across three architectures trained on the same corpus:

| method | effnet | ViT-B | ViT-L |
|---|---|---|---|
| hyperreenact | 0.6334 | 0.9293 | **0.9587** |
| sadtalker | 0.2757 | 0.4646 | **0.4161** |

hyperreenact rises 0.33 with backbone quality. SadTalker does not move, and
ViT-L is marginally *worse* than ViT-B. More backbone buys nothing here.

#### Why: the frames are barely generated

SadTalker animates **one still image** from audio. Measured by
`machine-learning/temporal_v2.py` over 100 test videos per method, as the
per-pixel standard deviation across 8 aligned crops of the same video:

| method | within-video pixel σ | share of that variation in the mouth |
|---|---|---|
| **sadtalker** | **2.99** | **0.159** |
| hyperreenact | 6.94 | 0.124 |
| cdf_synthesis | 12.64 | 0.109 |
| fomm | 15.04 | 0.080 |
| Deepfakes | 17.08 | 0.080 |
| **genuine Celeb-DF** | **16.31** | — |
| simswap | 19.78 | 0.105 |
| lia | 21.32 | 0.060 |

SadTalker's frames vary **5.5x less** than genuine video, and 16% of the little
variation there is sits in the mouth — against a mouth region that is 11.7% of
the crop. The face is frozen and the lips move. Per frame, the crop is almost
entirely unmodified pixels from the source still, so there is nothing for a
frame-level detector to find. That is the mechanism, and it explains the flat
score distribution exactly.

#### The cue that does work is temporal, and we throw it away

The same scalar — within-video pixel σ, one number per video, no network —
separates SadTalker from genuine Celeb-DF video at **AUC 0.9917**, on the exact
method where the ViT-L detector scores 0.4161.

Controls, because a cheap feature scoring 0.99 deserves suspicion:

* *Sampling interval.* SadTalker's sampled frames are closer together in the
  source video (median index gap 19 vs 30), which would deflate σ on its own.
  Restricting the reals to the same gap window (17–20) leaves their median σ at
  15.11 against SadTalker's 2.99 — **AUC 0.9620**.
* *Interval-free statistic.* The mean absolute difference between the two
  closest frames held, which has no dependence on how widely the video was
  sampled, gives **AUC 0.9476**.
* *Specificity.* It is not a general fake detector and must not be used as one.
  Against genuine video the same feature scores 0.44 on mcnet, 0.37 on simswap,
  0.32 on lia — most manipulations are *more* variable than real footage, not
  less. Only hyperreenact (0.857), the other still-ish render, comes near
  SadTalker.

So this is a second, orthogonal detector for one failure mode, not a feature to
blend into the existing score. It belongs as a separate signal at the video
level, where `video_processor` already has every frame it needs.

#### What is left

`simswap` 0.8187 and `inswap` 0.7307 on ViT-L (up from 0.6742 / 0.6803 on
ViT-B) are now the weakest *swap* results, and swaps are what someone is most
likely to actually run. They do respond to backbone scale, unlike SadTalker.

### 5b. Calibration collapses at JPEG q10

AUC degrades gracefully (0.82 -> 0.69) but real FPR blows out 0.086 -> 0.493.
The score distribution shifts bodily rather than losing separability. Cause is
in the config: `heavy` augmentation samples
`ImageCompression(quality_range=(30, 95))`, so q10 was never seen.

A `heavy_wide` augmentation level already exists for this test (JPEG floor
30->12, second pass 40->20, downscale floor 0.35->0.20). `heavy` was left
untouched so the not-yet-run `clipb_light_aug` ablation stays comparable.

## 6. Bugs fixed — do not reintroduce

### 6a. Session 3: the SBI sampler skewed the class balance

`dataset_v2.balanced_weights` takes `sbi_prob` and raises the real sampling
mass to `1 / (2 * (1 - p))`, so the split the model trains on is 50/50 *after*
SBI relabels sampled reals as pseudo-fakes. The original sampler aimed at 50/50
*before* SBI, so at `p = 0.5` training actually saw 25% real / 75% fake and the
model learned a fake-leaning prior — real FPR@0.5 of 0.459.

Verified empirically against the live manifest:

| `sbi_prob` | sampled real | post-SBI real | post-SBI fake | SBI share |
|---|---|---|---|---|
| 0.00 | 0.496 | 0.496 | 0.503 | — |
| 0.25 | 0.666 | 0.501 | 0.499 | 0.165 |

All 13 held-in manipulations draw within +/-5% of each other. `p >= 0.45` now
raises instead of silently skewing. Default `--sbi-prob` is **0.25**.

Fixing it moved held-out AUC 0.6615 -> 0.6715 (i.e. not at all — AUC is
threshold-free) while halving FPR. It was an operating-point bug, not a
discrimination bug.

### 6b. Session 4: four manipulations were downloaded and never ingested

`build_dataset_v2.do_df40` matched DF40 zips against a list of internal
layouts and silently ignored any member that fitted none of them. Four zips use
layouts that were not on the list:

| zip | layout | crops dropped |
|---|---|---|
| SiT | `<m>/{cdf,ff}/<bucket>/<video>/<n>.png` and `<m>/ff/<id>/<n>.png` | 31,885 |
| StyleGAN3 | the same | 31,885 |
| deepfacelab | `<m>/{real,fake}/frames/<video>/<n>.png` | 4,669 |
| heygen_new | the same, for the fakes only | 1,605 |

70,044 crops and four held-out manipulations, all present in `data/dl`, none in
the manifest — and nothing said so, because matching zero members and matching
every member looked identical from outside. Three more patterns now cover them,
and **anything matching no pattern is counted and printed**, which is the part
that stops this recurring.

`--merge` was added at the same time: `--only <zip>` previously wrote a
manifest containing nothing but the zips it rebuilt, which makes an incremental
rebuild silently destructive. Rows are keyed by path on write, so re-running a
zip replaces its own rows.

Two of the DF40 test zips also ship **zero-length members** (MidJourney 5,
starganv2 4). Extracted, they become empty files that `cv2.imread` returns
`None` for, which surfaces hours later as a `FileNotFoundError` from inside a
dataloader worker, naming a file that plainly exists. Members with
`file_size == 0` are now skipped at build time.

### 6c. Session 4: five held-out methods were never scored

Splitting is by hashed identity, and the flat image sets (stargan, styleclip,
MidJourney, whichfaceisreal, starganv2) have no video structure — one bucket
holds the whole method, so each one hashes into a single split. For all five
that split was not `test`, so they appeared in no per-method table. They were
in the manifest, counted in its totals, and never evaluated.

`--heldout-all-splits` (on `evaluate_v2.py`, `diagnose_v2.py` and
`report_v2.py`) draws held-out **fakes** from every split. That is sound
because held-out methods are never trained on; the reals stay inside the
evaluation split, because those *are* trained on and widening them would be
leakage.

It immediately found a failure larger than SadTalker's: **MidJourney scores
0.318 AUC** on the ViT-L model, with a median score of 0.047 against a real
median of 0.091. whichfaceisreal is 0.797. Both were invisible for the whole
project.

## 7. Code map (`machine-learning/`)

| File | Role |
|---|---|
| `protocol.py` | held-in / val-unseen / held-out method partition; generation-family map |
| `download_v2.py` | fetches the three corpora into `data/dl` |
| `build_dataset_v2.py` | unpacks zips into `data/processed_v2` + manifest; idempotent |
| `augment.py` | degradation augmentation (`none`/`light`/`heavy`/`heavy_wide`) + corruption sweep |
| `sbi.py` | Self-Blended Images pseudo-fake synthesis |
| `models_v2.py` | CLIP/DINOv2/DINOv3/ConvNeXt backbones, LN-tuning, hyperspherical head |
| `dataset_v2.py` | manifest dataset, SBI-aware balanced sampler, SBI hook |
| `train_v2.py` | training loop; selects on val-unseen macro AUC |
| `evaluate_v2.py` | per-method / per-dataset / per-corruption report; reads v1 *and* v2 checkpoints |
| `diagnose_v2.py` | per-frame score distributions: tells a *flat* method from an *inverted* one, per method and per generation family |
| `calibrate_v2.py` | picks the operating point on val against a false-positive budget, under deployment-like degradation, and writes it into the checkpoint config |
| `temporal_v2.py` | within-video pixel variation per method — the still-driven-render signal the frame-level model cannot see |
| `aggregate_v2.py` | compares video-level aggregators on ranking *and* on recall at a matched false-positive rate |
| `rank_runs.py` | ranks finished evaluations by held-out AUC, with the method count each average covers |
| `finalize_v2.sh` | the post-sweep pass: re-score everything on one manifest, rank, full corruption sweep, calibrate, ensemble |
| `report_v2.py` | aggregates evals into `runs/v2_comparison.md`; also scores ensembles (mean + rank-average) |
| `run_experiments.sh` | the sweep; stages `core` / `ablations` / `large` / `all` |

`inference.py` dispatches on `config["version"]`, so `load_model` / `predict`
work unchanged for both checkpoint generations.

### Backend change (uncommitted)

`backend/services/video_processor.crop_face` gained an `aligned` mode
reproducing the training crop: a square of side `1.4 * max(box_w, box_h)`
centred on the MTCNN box centre, raised by `0.10 * side`. Constants were fitted
by matching MTCNN boxes on the raw FF++ videos against the corresponding
DeepfakeBench crops — median normalized cross-correlation 0.79 over 25 videos.
`detector.crop_style()` reads the loaded checkpoint's version, so v1 keeps the
old fixed-20px-margin crop and nothing breaks while v2 is unfinished.

## 8. Timings (measured wall clock, train+eval)

| run | per epoch | total |
|---|---|---|
| `effnet_data_only` | 108 s | 10 min |
| `clipb_full` | 237 s | 24 min |
| `clipl_full` | 965 s | ~96 min |

Evaluation over ~27k frames x 4 corruptions: ~1 min (EfficientNet), ~3 min
(ViT-B), ~10 min (ViT-L). Dataloader does 475 img/s at 8 workers — never the
bottleneck. GPU is an RTX 4060 Laptop, 8 GB; ViT-L at batch 24 peaks ~6.0 GB.

## 9. Resume

`run_experiments.sh` skips any run whose log already ends in `checkpoint: `, so
re-running the same command picks up where it stopped.

```bash
cd /home/mudit/deepfake-detection

# 1. Redo the one interrupted evaluation (~10 min).
.venv/bin/python machine-learning/evaluate_v2.py \
    --checkpoint machine-learning/checkpoints/v2_clip_vit_l14_ln_clipl_full_20260921_225402 \
    --data data/processed_v2 --split test --amp \
    --max-per-method 800 --max-real 4000 \
    --corruptions clean jpeg_q40 jpeg_q10 downscale_0.25

# 2. Resume the sweep — skips the 3 completed runs. ~3h40m remaining:
#    5 ViT-B ablations (~120 min), dinov3l_full (~70 min), convnext_full (~30 min).
bash machine-learning/run_experiments.sh all

# 3. Comparison table + ensemble check.
.venv/bin/python machine-learning/report_v2.py
.venv/bin/python machine-learning/report_v2.py --amp \
    --ensemble machine-learning/checkpoints/v2_clip_vit_l14_ln_clipl_full_* \
               machine-learning/checkpoints/v2_clip_vit_b16_ln_clipb_full_*
```

The queued ablations are `clipb_no_sbi`, `clipb_light_aug`,
`clipb_linear_probe`, `clipb_mlp_head`, `clipb_no_latent`. The last two are the
lowest-information runs in the set (head shape and latent augmentation are
second-order next to backbone, SBI and augmentation strength) — drop them to
save ~50 min if time matters.

### Then, in value order

1. **Diagnose SadTalker (§5a).** The largest single defect and possibly a whole
   invisible class of generation. Start with its score histogram versus reals,
   then check `heygen` and `deepfacelab`. If it is a class failure, add a
   held-in talking-head method rather than scaling the backbone.
2. **`heavy_wide` retrain of the winner** (~90 min) to test whether the q10
   calibration collapse (§5b) is purely a train/test augmentation-range gap.
   The level exists; pass `--augment heavy_wide`.
3. **Threshold selection.** Every number here is at the arbitrary 0.5 cutoff.
   Pick the operating point on the *val* split against a stated false-positive
   budget, write it into the checkpoint config, and have `backend/config.py`'s
   risk bands read it instead of the v1 constants.
4. **Full 12-corruption sweep on the finalists** (~30 min) — drop
   `--corruptions` for the complete table.
5. **Ship it.** Point `MODEL_DIR` in `backend/.env` at the winning checkpoint;
   `detector.crop_style()` switches the crop convention automatically. Run
   `backend/smoke_test.py` before and after.

## 10. Open items

* **Nothing is committed.** `git status` shows 3 modified files
  (`backend/services/detector.py`, `backend/services/video_processor.py`,
  `machine-learning/inference.py`) and 11 new ones. `data/` and `logs/` are
  gitignored. The working tree is the only copy — commit early next session.
* `Deepfake-Eval-2024` (`nuriachandra/Deepfake-Eval-2024`) is the one genuinely
  in-the-wild 2024 benchmark on the Hub. Manual-approval gated, not downloaded.
  Worth requesting — a better final test than anything currently held out.
* `evaluate_v2.py` aggregates video-level scores with a plain mean, matching
  `video_processor.aggregate`. A trimmed or top-k mean is likely better and has
  not been tried.
* No ensembling has been measured yet; `report_v2.py --ensemble` is written but
  unrun.
* The v1 `Deepfakes` manipulation is held-in, which keeps the v1-vs-v2
  comparison fair but means the v1 model's 0.999 and v2's held-out numbers are
  not measuring the same thing. Compare v1 and v2 on the held-out set only.
