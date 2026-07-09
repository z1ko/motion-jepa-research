
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
