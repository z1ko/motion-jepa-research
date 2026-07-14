# CARE-PD linear-probe eval is blind to encoder training quality

## Observation

Every v16-19 training variant (loss, masking, schedule, epoch count) lands in the same narrow band on CARE-PD-BMCLab (748 walks, UPDRS-gait severity, 3 classes):

| checkpoint | loss | epochs | walk F1 |
|---|---|---|---|
| version_16 | centered_ce | 200 | 0.518 |
| version_17 | smooth_l1 | 200 | 0.529 |
| version_18 | smooth_l1 | 400 | 0.539 |
| version_19 | smooth_l1 | 200 | 0.544 |

0.02 spread across meaningfully different runs — every lever pulled moved this less than run-to-run noise.

## Test: is training contributing anything?

Compared against a random-init encoder and raw input (no encoder, masked mean-pool over time):

| representation | walk F1 |
|---|---|
| Raw input (no encoder) | 0.466 |
| Random-init encoder | 0.502 – 0.552 |
| **Trained (v16–19)** | **0.518 – 0.544** |

Trained doesn't beat random-init. An untrained transformer matches or beats every checkpoint produced so far.

## Ruled out: pooling choice

| pooling | trained (v16) | random-init |
|---|---|---|
| mean | 0.518 | 0.502 |
| per_group | 0.522 | 0.526 |
| per_segment | 0.531 | 0.504 |
| max | 0.481 | 0.564 |

No pooling variant separates trained from random (`max` favors random). Not a pooling artifact.

## Diagnosis

Task/probe ceiling, not representation quality:
- Mean/max/per-axis pool + linear probe reduces each window to low-order stats (avg posture, amplitude, speed).
- BMCLab's UPDRS-gait label correlates directly with those — raw signal average (0.466) already recovers most of the separable signal.
- A random-init transformer (residual + LayerNorm) is close to identity at init, so it preserves that signal just as well as anything trained.
- Whatever JEPA actually learns (temporal structure, cross-joint relations) is either not captured by this label, or discarded by pool-then-probe before the classifier sees it.

**BMCLab linear-probe numbers can't be used to compare training runs.**

## What's still valid

Training-time diagnostics (`collapse/*`, `distillation/*`, `alignment/teacher_student_cosine_sim`, `grad/norm`) read the encoder's own representations directly, not through the lossy pool-and-probe step — still trustworthy.

## Correction: blindness is CARE-PD-specific, not a broken eval

Same trained-vs-random test on SOMA/HumanEva (action) and DanceDB (emotion) — not CARE-PD, not subject-constant labels:

| dataset | label | random | trained (v16) | delta | majority |
|---|---|---|---|---|---|
| SOMA | action (11 cls) | 0.134 | 0.376 | **+0.242** | 0.087 |
| HumanEva | action (6 cls) | 0.624 | 0.829 | **+0.205** | 0.215 |
| DanceDB | emotion (13 cls) | 0.151 | 0.232 | +0.081 | 0.052 |

Trained clearly beats random here. JEPA pretraining learns real, transferable structure — the eval pipeline detects it when the task allows it. The random≈trained result is specific to CARE-PD-BMCLab (likely 3DGait too), not a broken methodology. SOMA/HumanEva are the working instrument for testing pretraining changes (masking ratio, loss, architecture) — check there, not CARE-PD.

## Options going forward

1. Re-run `--compare-pooling` on 3DGait too (not yet done).
2. Use SOMA/HumanEva (and DanceDB, weaker) as the real quality instrument.
3. Stop using CARE-PD linear-probe deltas to pick between runs.

## What CARE-PD's own paper does differently

Read their eval code ([TaatiTeam/CARE-PD](https://github.com/TaatiTeam/CARE-PD)) for MotionBERT/MotionCLIP/PoseFormerV2/POTR/MixSTE/MotionAGFormer/MoMask:
- Frozen backbone + linear head, but head trained via AdamW with per-backbone Optuna-tuned hyperparameters, not one fixed `sklearn.LogisticRegression(C=1.0)`.
- Class-weighted/focal loss, never plain unweighted CE.
- Per-video logit averaging baked into training/model-selection, not bolted on after like our walk-level scoring.
- Demographics (`--medication`/`--metadata`) exist but default off.
- A separate `end2end` fine-tune mode exists, reported as an ablation on PD-GaM only.

Tested the one portable piece (class-weighting) on our v16/BMCLab:

| | window F1 | walk F1 |
|---|---|---|
| unweighted | 0.627 | 0.518 |
| `class_weight='balanced'` | 0.632 | 0.527 |

+0.009 walk F1 — negligible, rules out class weighting. Consistent with `walk_majority_baseline_mean` (23.8%) sitting *below* chance (33%): BMCLab's real problem is severity being close to constant-per-subject, not class imbalance — no in-fold reweighting fixes that.

**The difference that isn't portable**: every non-MoMask/MotionCLIP backbone is pretrained on Human3.6M (large-scale, real video, hundreds of subjects) before ever seeing CARE-PD; MoMask/MotionCLIP pretrain on HumanML3D/AMASS using an engineered 263-dim feature (root velocity, local joint pos/rot/vel, foot contact), not raw joint angles. CARE-PD is only their downstream probe set — never the pretraining set. Our encoder has to build everything from this project's own (much smaller) corpus alone.

## Options to recover signal, ranked

1. **Not worth it**: replicate their gradient-descent head / WCE-focal / hyperparameter search — tested the core idea (class weighting), sub-1% effect.
2. **High-effort, most likely to work**: pretrain on a much larger motion corpus before CARE-PD, matching what every one of their backbones already has.
3. **Match their engineered HumanML3D-style feature** instead of raw pos/vel/acc/tau — moderate effort, testable on the current architecture.
4. **Cheap sanity check first**: try their `end2end` fine-tune mode on our encoder + BMCLab. If full fine-tuning helps but frozen linear probing can't see it, that points at option 3 over option 2.

## Ruled out: torque as a subject-scaling confound

Hypothesis: `tau` (joint torque) leaks per-subject OpenSim scaling (mass/height/segment scale) rather than gait signal. Tested by zeroing `tau`:

| representation | with torque | torque zeroed |
|---|---|---|
| Raw input | 0.466 | 0.355 (−0.111) |
| Random-init | 0.50–0.56 | 0.505 |
| Trained (v16) | 0.518 | 0.469 (−0.049) |

Rejected — removing torque makes everything worse, so it carries real signal, not confound. Trained's smaller relative drop (−9% vs raw's −24%) is a mild positive sign but doesn't change the core finding: trained and random stay statistically indistinguishable either way. Keep torque in the input.

## Ruled out: mean-pooling destroying signal

CARE-PD's own backbones also use plain mean/flatten pooling (checked `model/motion_encoder.py`), so this tested a real hypothesis, not "catching up." Built an attentive probe (`AttentiveProbeHead`/`run_attentive_probe`, `--attentive-probe`): learned query cross-attends over every unpooled token, trained per LOSO fold.

| | mean-pool | attentive | delta |
|---|---|---|---|
| Trained (v16) | 0.518 | 0.545 | +0.027 |
| Random-init | 0.480 | 0.507 | +0.027 |

Rejected — identical delta whether trained or random; attentive pooling just adds probe capacity regardless of what the encoder learned (same pattern as the MLP probe). Mean-pooling wasn't the bottleneck. `--attentive-probe` stays as a permanent opt-in diagnostic, not a replacement for mean-pool reporting.

## Ruled out: DMU (deviation-from-mean-unimpaired)

`papers/GaitEncoder.pdf`'s continuous score instead of a classifier: Mahalanobis distance from a reference ("unimpaired") class's latent mean/variance, refit per LOSO fold. Ported as `run_dmu_probe`/`--dmu-probe`.

Trained (v32) vs. random-init, Spearman r vs. severity:

| dataset | trained window r | random window r | trained walk r | random walk r |
|---|---|---|---|---|
| CARE-PD-BMCLab | 0.731 | 0.621 | 0.452 | 0.329 |
| CARE-PD-PD-GaM | 0.464 | 0.563 | −0.057 | 0.243 |
| CARE-PD-3DGait | 0.267 | 0.295 | 0.235 | 0.219 |

Rejected, worse than expected on PD-GaM (random beats trained at walk level: 0.243 vs. −0.057). Same root cause as the classifier: an untrained near-identity transformer already preserves the low-order per-subject statistics severity is confounded with, and Mahalanobis-from-centroid is exactly that kind of low-order statistic. A continuous score doesn't escape a label confound. Stays as an opt-in diagnostic (`--dmu-probe`), not evidence of training quality on BMCLab/PD-GaM.

## Update: blindness confirmed across the full v2 hyperparameter sweep

Extended the same trained-vs-random test across this session's 5-version sweep on the current architecture (`runs/v2/version_0..4`: encoder/predictor depth, dropout, LR, weight decay, EMA momentum each varied). Numbers from `runs/eval_results.parquet` (`motion_jepa/evaluation/results_table.py`; every eval CLI run upserts a row, `scripts/backfill_eval_results.py` ingested history). Delta = trained − random, window/walk f1:

| dataset | delta range across sweep | pattern |
|---|---|---|
| CARE-PD-3DGait | −.066 to +.042 | negative or ~zero in 6/7 checkpoints |
| CARE-PD-BMCLab | −.037 to +.049 | straddles zero; only positive outlier is the sweep's most-degraded checkpoint (val_loss +44% off its minimum) |
| CARE-PD-PD-GaM | +.001 to +.104 | consistently positive but small, and often below its own majority baseline |

No relationship to depth, LR, dropout, or EMA momentum on any of the three. Meanwhile the *same checkpoints* show a clean, large signal on BABEL (delta +.04 up to +.118, tracking architecture/LR/dropout changes). Same asymmetry as the original SOMA/HumanEva-vs-BMCLab finding, now confirmed on a different architecture and a much wider set of training variants. **Conclusion unchanged, on stronger evidence: use BABEL/HumanEva deltas to compare checkpoints, not CARE-PD, on any architecture tried so far.**
