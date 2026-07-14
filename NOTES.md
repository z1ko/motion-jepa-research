
Larger-window trade-off worse than "less data" — reintroduces exact bias you're trying to remove, different door. Why:

Coverage cost of going larger (train split, 15,718 trials, 19.9h total):

window	usable trials	frame coverage	windows (∝ overlap)
300 (3s)	62.4%	83.3%	86,241
400 (4s)	45.1%	70.2%	48,687
500 (5s)	32.3%	57.6%	29,327
600 (6s)	24.0%	47.7%	17,764
800 (8s)	17.4%	38.0%	7,767
1000 (10s)	14.4%	32.2%	2,454

Median trial length only 3.66s — raw corpus property, not fixable by picking bigger number.

Bigger problem: survival very uneven across datasets. Checked per-dataset survival rates each window size. At 600+ frames, WEIZMANN, EKUT, CNRS, DFaust, SSM, Transitions almost entirely excluded (0-6% trials survive), while HUMAN4D, Eyes_Japan_Dataset, HDM05, PosePrior, SFU stay ~85-100% intact. Large-window training set stops being "whole corpus, just windowed longer", becomes "whichever datasets happen to record long continuous trials." Exactly dataset-composition confound you started this conversation to eliminate — just moved from coordinate channels into which trials get selected.

Recommendation: don't chase window size isolation. Given length distribution, ~400-500 frames (4-5s) close to practical ceiling for good frame coverage (57-70%) + even dataset representation (18/19 datasets meaningfully present). Beyond that: not training on "more motion" broadly, training on narrower slice of datasets with long recordings.

Q: so CARE-PD is at the moment not usable as a stable evaluation metric, so our model is not enough for that dataset?

A: Not "model insufficient" — evidence points the other way. Same architecture, same probe, same session: HumanEva/BABEL show trained clearly beating random (+14 to +23pts). If the encoder were fundamentally too weak, that gap wouldn't exist anywhere. It's CARE-PD specifically that's blind, not the model in general.

Per CARE-PD-REPORT.md's earlier diagnosis: raw input (no encoder at all, just averaged over time) already scores ~0.466 walk F1 on BMCLab, barely below random-init (~0.5-0.56) and trained (~0.52-0.61). That means the label correlates with coarse, low-order kinematic stats (average posture/speed) that survive any reasonably-initialized transformer — residual connections + LayerNorm make a random-init model close to an identity map, so it doesn't destroy that signal, and training doesn't obviously add much on top of it either, at least not in a way mean/max-pool + linear-probe can see. Already ruled out that this is a pooling artifact — same story across mean/per_group/per_segment/max.

So the honest framing: CARE-PD's severity labels sit at a ceiling this evaluation protocol can't see past, not proof the encoder has nothing to offer. Two different explanations remain open, and today's data can't tell them apart yet:

The JEPA pretraining genuinely isn't learning anything beyond what raw statistics give you (possible, but contradicted by HumanEva/BABEL working).
It's learning real structure, but severity-from-gait needs it fed through something richer than frozen-mean-pool-then-linear-classifier — CARE-PD's own paper trains their classifier head with AdamW/class-weighted loss/per-video logit averaging during training itself, not a one-shot sklearn LogisticRegression on top of frozen features (see the note in CARE-PD-REPORT.md's last section).
Practically: don't chase CARE-PD improvements right now — use HumanEva/BABEL to judge training changes, they're the instrument that actually responds. If CARE-PD matters for the end goal (clinical severity), the next real step isn't more pretraining, it's changing the eval protocol itself (end-to-end fine-tune, or a trained probe head like the paper's), not something the ema_momentum/LR/dropout sweep can fix.