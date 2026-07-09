
The larger-window trade-off is worse than just "less data" — it also reintroduces a version of the exact bias you're trying to remove, just through a different door. Here's why.

Coverage cost of going larger (train split, 15,718 trials, 19.9h total):

window	usable trials	frame coverage	windows (∝ overlap)
300 (3s)	62.4%	83.3%	86,241
400 (4s)	45.1%	70.2%	48,687
500 (5s)	32.3%	57.6%	29,327
600 (6s)	24.0%	47.7%	17,764
800 (8s)	17.4%	38.0%	7,767
1000 (10s)	14.4%	32.2%	2,454

Median trial length is only 3.66s — that's a property of the raw corpus, not something the pipeline can fix by picking a bigger number.

The bigger problem: survival becomes very uneven across datasets. I checked per-dataset survival rates at each window size. At 600+ frames, datasets like WEIZMANN, EKUT, CNRS, DFaust, SSM, Transitions are almost entirely excluded (0-6% of their trials survive), while HUMAN4D, Eyes_Japan_Dataset, HDM05, PosePrior, SFU stay ~85-100% intact. That means a large-window training set stops being "the whole corpus, just windowed longer" and becomes "whichever datasets happen to record long continuous trials." That's exactly the kind of dataset-composition confound you started this conversation trying to eliminate — just moved from the coordinate channels into which trials get selected.

Recommendation: don't chase window size in isolation. Given the length distribution, ~400-500 frames (4-5s) is close to the practical ceiling for keeping both good frame coverage (57-70%) and even dataset representation (18/19 datasets meaningfully present). Beyond that you're not really training on "more motion" broadly, you're training on a narrower slice of datasets that happen to have long recordings.