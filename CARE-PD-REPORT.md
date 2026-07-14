# CARE-PD linear-probe eval is blind to encoder training quality

## The observation

Every training variant this project has tried on CARE-PD-BMCLab (748 walks, UPDRS-gait severity, 3 classes) lands in the same narrow band, regardless of loss function, masking strategy, schedule fix, or epoch count:

| checkpoint | loss | epochs | walk F1 (linear probe) |
|---|---|---|---|
| version_16 | centered_ce | 200 | 0.518 |
| version_17 | smooth_l1 | 200 | 0.529 |
| version_18 | smooth_l1 | 400 | 0.539 |
| version_19 | smooth_l1 | 200 | 0.544 |

A 0.02 spread across four meaningfully different training runs. Every other lever pulled this session (loss redesign, per-sample mask diversity, LR schedule fix, EMA hook fix, 2x epoch budget) moved this number by less than run-to-run noise.

## The test

If different training runs all land in the same band, the natural question is whether *any* training is contributing anything at all. Tested two baselines against the same eval pipeline, same BMCLab data, same probes:

1. **Random-init encoder** — `MotionJEPAModule(config)` constructed fresh, teacher encoder used *without ever loading a checkpoint*. Same architecture, random weights.
2. **Raw input, no encoder** — window's raw (T, 49 DOF, 4 channel) tensor, masked mean-pooled over time only, fed straight to the linear probe.

| representation | walk F1 |
|---|---|
| Raw input (no encoder) | 0.466 |
| Random-init encoder (mean pool) | 0.502 – 0.552 (run-to-run, untrained weights) |
| **Trained encoders (version_16–19)** | **0.518 – 0.544** |

Trained encoders do not beat a randomly-initialized one. An untrained transformer beats or matches every checkpoint this project has produced.

## Ruled out: pooling choice

Repeated the comparison across all four pooling strategies (`mean`, `per_group`, `per_segment`, `max`) in case mean-pooling specifically was washing out a real trained/random gap:

| pooling | trained (v16) walk F1 | random-init walk F1 |
|---|---|---|
| mean | 0.518 | 0.502 |
| per_group | 0.522 | 0.526 |
| per_segment | 0.531 | 0.504 |
| max | 0.481 | 0.564 |

No pooling variant separates trained from random. `max` pooling actually favors the random encoder. This isn't a pooling artifact — it holds across every axis-collapse strategy available.

## Diagnosis

The eval ceiling is set by the **task and probe protocol**, not by representation quality:

- Mean/max/per-axis pooling followed by a linear (or kNN/MLP) probe reduces each window to low-order statistics of the 49-DOF trajectory (average posture, amplitude, speed-like quantities).
- BMCLab's UPDRS-gait label correlates directly with those low-order statistics — a plain average of the raw signal (0.466) already recovers most of the separable signal.
- A transformer at random init, thanks to residual connections + LayerNorm, is close to an information-preserving (near-identity-ish) map at initialization — it doesn't destroy that low-order signal, so it scores just as well as anything trained on top of it.
- Whatever the JEPA pretraining is actually learning (temporal structure, cross-joint relationships, higher-order kinematic patterns) is either (a) not being captured by this label, or (b) being discarded by the pool-then-linear-probe step before the classifier ever sees it.

**This means the linear-probe/BMCLab number cannot currently be used to judge whether one training run produced a better encoder than another.** It has been treated as the primary quality signal for several rounds of experiments this session; it isn't sensitive enough to distinguish them.

## What still is a valid signal

Training-time diagnostics logged this session (`collapse/*`, `distillation/*`, `alignment/teacher_student_cosine_sim`, `grad/norm`) don't route through this bottleneck — they're computed directly on the encoder's own representations during training, not through a lossy pool-and-probe. These remain trustworthy for comparing runs.

## Correction: the blindness is CARE-PD-specific, not a broken eval methodology

Everything above could be read as "our linear-probe eval can't see representation quality, period." Ran the identical trained-vs-random-init comparison (mean-pool, same `LogisticRegression` probe, same LOSO protocol) on the three other held-out eval sets already wired into the CLI — SOMA/HumanEva (action) and DanceDB (emotion), none of them CARE-PD, none of them subject-constant-label the way BMCLab's severity score is:

| dataset | label | random-init walk F1 | trained (v16) walk F1 | delta | majority baseline |
|---|---|---|---|---|---|
| SOMA | action (11 classes) | 0.134 | 0.376 | **+0.242** | 0.087 |
| HumanEva | action (6 classes) | 0.624 | 0.829 | **+0.205** | 0.215 |
| DanceDB | emotion (13 classes) | 0.151 | 0.232 | +0.081 | 0.052 |

This is a completely different picture from BMCLab. Trained beats random by 20+ points on SOMA and HumanEva, and clearly (if more modestly) on DanceDB. **The JEPA pretraining is learning real, transferable, linearly-decodable structure — the eval pipeline can detect it when the task allows it.** The random≈trained finding is specific to CARE-PD-BMCLab (very likely CARE-PD-3DGait too, not yet re-checked with this control), not evidence that training doesn't work or that the whole evaluation methodology is broken.

This also means the "make the pretext task harder" question has a real path to an answer now: SOMA/HumanEva give a working instrument. A masking-ratio (context/target) experiment that's invisible on CARE-PD could still show up as a widening or narrowing of the SOMA/HumanEva trained-vs-random gap — that's the metric to re-check, not CARE-PD walk F1, if that experiment gets run.

## Options going forward

1. **Re-run `--compare-pooling` baselines on 3DGait too** (not yet done — this report only covers BMCLab) before generalizing the conclusion across both CARE-PD cohorts.
2. **Use SOMA/HumanEva (and to a lesser extent DanceDB) as the actual representation-quality instrument going forward** — confirmed above to separate trained from random where CARE-PD-BMCLab cannot. Any pretraining change (masking ratio, loss, architecture) should be checked here first.
3. **Stop using CARE-PD linear-probe deltas to pick between training runs.** Use the training-time diagnostics and the SOMA/HumanEva/DanceDB probes instead — CARE-PD-BMCLab specifically is not a discriminating signal, independent of what training produces.

## What the CARE-PD paper's own code does differently

Read the actual eval pipeline for their SOTA baselines (MotionBERT, MotionCLIP, PoseFormerV2, POTR, MixSTE, MotionAGFormer, MoMask) from [TaatiTeam/CARE-PD](https://github.com/TaatiTeam/CARE-PD) — `model/motion_encoder.py`, `model/backbone_loader.py`, `train.py`, `const/const.py`, `run.py` — to check whether they hit the same representation-invariance wall and, if not, what's different.

**Their main-results protocol (`train_mode='classifier_only'`, the default in `run.py`) is:**

- Frozen backbone (`freeze_backbone()` in `MotionEncoder.__init__`) + a classifier head that is architecturally still linear (`classifier_hidden_dims: "no_hidden_layers"` in every one of their per-backbone hypertune configs) — so structurally the same "linear probe on frozen features" idea we're using.
- But the linear head is **trained by gradient descent (AdamW)**, not `sklearn.LogisticRegression` — with per-backbone hyperparameter search (`configs/best_configs_augmented/Hypertune/*/best_params.json`: lr, batch_size, epochs, weight_decay, all tuned via Optuna) rather than one fixed `C=1.0` default.
- **Class-imbalance-aware loss**: every backbone's tuned config uses `WCELoss` (class-weighted cross-entropy) or `FocalLoss` (`alpha`/`gamma` tunable), never plain unweighted CE (`learning/criterion.py`).
- **Per-video logit aggregation during training itself**, not just at eval time: `train_model`/`validate_model` in `train.py` average each video's per-clip logits (`torch.stack(predictions).mean(dim=0)`) to compute video-level accuracy every epoch, and `final_test` majority-votes per-clip predicted classes — closer to what we added as walk-level scoring, but baked into training/model-selection, not bolted on after.
- `--medication` and `--metadata` (age/sex/BMI/height/weight) flags exist and get concatenated straight into the classifier input in `MotionEncoder.forward`, but **default to off** (`default=0`, `default=''` in `run.py`) — their headline numbers are not leaking demographics through a side channel.
- A separate `train_mode='end2end'` mode exists (backbone gets its own `lr_backbone`, fine-tuned jointly with the head) but is tuned and reported as a distinct ablation on PD-GaM, not the main frozen-encoder benchmark.

**Tested the one cheap, directly-portable piece of this** (class-imbalance weighting) against our own trained v16 encoder on BMCLab: `LogisticRegression(class_weight='balanced')` vs the current unweighted default.

| | window F1 | walk F1 |
|---|---|---|
| unweighted (current) | 0.627 | 0.518 |
| `class_weight='balanced'` | 0.632 | 0.527 |

Negligible (+0.009 walk F1) — rules out class weighting as the fix. Consistent with the report's earlier finding: `walk_majority_baseline_mean` (23.8%) being *below* chance (33%) shows BMCLab's real problem isn't per-class imbalance, it's that severity is close to a **constant per subject** — LOSO folds hold out subjects whose label the training fold's class balance doesn't predict, which no amount of reweighting inside a fold fixes.

**The difference that actually matters and isn't portable by a config change**: every one of their non-MoMask/MotionCLIP backbones (MotionBERT, PoseFormerV2, MixSTE, MotionAGFormer, POTR) is a 2D-pose-lifting network **pretrained on Human3.6M** (large-scale, real video, hundreds of subjects) before ever seeing CARE-PD — `load_pretrained_backbone` in `backbone_loader.py` loads external checkpoints, it doesn't train these backbones from scratch on CARE-PD data. MoMask/MotionCLIP are pretrained on **HumanML3D/AMASS** (44-hour text-to-motion corpus) using the engineered 263-dim HumanML3D feature (root velocity + local joint position/rotation/velocity + foot contact — see `data/preprocessing/smpl2humanml3d.py`, `assets/stats/HumanML3D_norm_data/`), not raw joint angles. Their "linear probe" numbers reflect linearly reading out features from encoders that already saw orders of magnitude more motion data than CARE-PD itself contains — CARE-PD is *only* the downstream probe set for them, never the pretraining set. Our encoder is trained on this project's own corpus and has to build any generalizable representation from that alone.

## Options to actually recover signal, ranked by expected effect

1. **Not worth doing**: replicate their exact gradient-descent classifier head / WCE-Focal loss / hyperparameter search. Tested the core idea (class weighting) directly against our data — sub-1% effect. The gap is not the probe.
2. **Structural, high-effort, most likely to actually work**: give the encoder more/richer pretraining data before this small-corpus downstream task, mirroring what every one of their backbones already has — e.g. pretrain (or continue pretraining) on a much larger motion corpus than what's currently in `data/processed/motion` before ever touching CARE-PD, so there's more than this project's own data for JEPA to build structure from.
3. **Match their HumanML3D-style engineered feature instead of raw pos/vel/acc/tau** as the model's input/target space — a canonical, velocity-and-local-frame representation may be more linearly decodable downstream than raw generalized coordinates, independent of any pretraining-data-scale fix. Moderate effort (new preprocessing path), testable on the existing encoder architecture.
4. **Cheap sanity check worth running before either of the above**: their `end2end` (fine-tune backbone + head jointly, small `lr_backbone`) mode on our own encoder + BMCLab. If full fine-tuning *can* pull the encoder toward a useful representation but frozen linear probing can't see it, that confirms the encoder has usable structure the linear probe protocol just can't extract — pointing at option 3 (representation/feature space) over option 2 (more pretraining data) as the faster fix.

## Ruled out: torque as a subject-scaling confound

Hypothesis: our input's `tau` (joint-torque, from inverse dynamics) channel might be leaking per-subject OpenSim scaling artifacts (`subject_mass_kg`, `subject_height_m`, per-segment `_scale_x/y/z` — all subject-constant, baked into `utils.py`'s preprocessing) rather than genuine gait-quality signal, and that this could be part of what makes BMCLab look subject-identity-dominated. Tested by zeroing the `tau` channel at the model's input and re-running the same three probes:

| representation | walk F1, with torque | walk F1, torque zeroed |
|---|---|---|
| Raw input (no encoder) | 0.466 | 0.355 (−0.111) |
| Random-init encoder | 0.50–0.56 (run-to-run) | 0.505 |
| Trained encoder (v16) | 0.518 | 0.469 (−0.049) |

Hypothesis rejected. Removing torque makes every representation *worse*, not better — torque carries real, linearly-decodable signal for this task, not confound noise. Notably the trained encoder's relative drop (−9%) is smaller than the raw baseline's (−24%), a mild positive signal that the encoder isn't just passing the torque channel through unchanged — but this doesn't move the core finding: trained (0.518) and random-init (0.50–0.56) are still statistically indistinguishable, with or without torque. Don't drop torque from the input; if anything it should stay.

## Ruled out: mean-pooling destroying signal

CARE-PD's own backbones all use plain mean/flatten pooling too (checked `model/motion_encoder.py` — no attention pooling anywhere in their repo), so this wasn't "catching up to them," it was testing a real independent hypothesis: is our fixed mean-pool-before-classify step discarding information the encoder actually captured?

Implemented an attentive probe (`AttentiveProbeHead`/`run_attentive_probe` in `motion_jepa/evaluation/probes.py`, `compute_token_embeddings` in `encoder.py`, `--attentive-probe` flag on the `evaluate` CLI): a learned query cross-attends over every unpooled token, trained fresh per LOSO fold by gradient descent, instead of averaging tokens before the classifier ever sees them.

| | mean-pool walk F1 | attentive walk F1 | delta |
|---|---|---|---|
| Trained (v16) | 0.518 | 0.545 | +0.027 |
| Random-init | 0.480 | 0.507 | +0.027 |

Hypothesis rejected. The delta is identical (+0.027) whether the encoder is trained or randomly initialized — attentive pooling adds probe capacity, which helps a little regardless of what the encoder learned, the same pattern already seen with the MLP probe. Trained still sits ~0.03–0.04 above random at both pooling levels, well inside the noise band already established across runs (0.50–0.56). Mean-pooling was not the bottleneck; don't spend effort replacing it as a fix for this specific problem. The attentive probe is now a permanent extra diagnostic in the eval CLI (opt-in, `--attentive-probe`), not a replacement for the mean-pool numbers used for CARE-PD-comparable reporting.

## Ruled out: DMU (deviation-from-mean-unimpaired) as an escape from the classifier's blind spot

`papers/GaitEncoder.pdf` scores gait impairment with a continuous, unsupervised-at-scoring-time metric instead of a classifier: Mahalanobis distance from a reference ("unimpaired") class's latent mean/variance. Ported as `run_dmu_probe`/`--dmu-probe` in `motion_jepa/evaluation/probes.py`+`cli.py` (diagonal covariance, refit per LOSO fold from the training split's lowest-label class, correlation pooled across all folds against the ordinal severity label — see the plan's rationale for why per-fold correlation would be degenerate here). The hope: a continuous distance might pick up structure a hard 3-class decision boundary washes out.

Trained (v32) vs. random-init teacher encoder, Spearman r between DMU and severity label:

| dataset | trained window r | random window r | trained walk r | random walk r |
|---|---|---|---|---|
| CARE-PD-BMCLab | 0.731 | 0.621 | 0.452 | 0.329 |
| CARE-PD-PD-GaM | 0.464 | 0.563 | −0.057 | 0.243 |
| CARE-PD-3DGait | 0.267 | 0.295 | 0.235 | 0.219 |

Hypothesis rejected, and worse than expected: on PD-GaM the random-init encoder's DMU correlation is *higher* than the trained encoder's (walk-level: 0.243 vs. −0.057, i.e. trained shows no relationship at all at walk level while random does). BMCLab and 3DGait show trained slightly ahead, but well inside the noise band this report has already established for classifier probes (random ties or beats trained is the norm here, not the exception). DMU inherits the same root cause as the linear probe (see Diagnosis above): an untrained near-identity transformer already preserves the low-order per-subject statistics that severity is confounded with, and Mahalanobis distance from a class centroid is exactly the kind of low-order statistic that survives no training at all. A continuous distance metric doesn't escape a confound that's about *what the label correlates with*, only about *how the score is computed*.

DMU stays in the eval CLI as an opt-in diagnostic (`--dmu-probe`) since it's cheap (reuses embeddings already computed for the linear probe) and may still be informative on a dataset where severity isn't subject-confounded — but do not use it as evidence that training helped on CARE-PD-BMCLab or CARE-PD-PD-GaM specifically.
