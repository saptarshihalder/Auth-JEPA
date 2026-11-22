import numpy as np


def calibrate(val_scores):  # per expert and per file format, on real validation photos
    med = np.median(val_scores)
    mad = np.median(np.abs(val_scores - med)) * 1.4826
    return lambda s: (s - med) / mad


def council(z_a, z_b):  # model A (13 cameras) and model B (+ 25 cameras): keep the less suspicious score
    return np.minimum(z_a, z_b)


def threshold(z_seen_real):  # cut-off: 5% of real photos from known cameras are flagged
    return np.quantile(z_seen_real, 0.95)


def is_ai(z, thr):  # flag as AI if above the cut-off
    return z > thr
