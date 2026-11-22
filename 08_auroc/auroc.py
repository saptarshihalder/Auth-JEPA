import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(real_scores, fake_scores):  # 1.0 = perfect, 0.5 = guessing
    y = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
    return float(roc_auc_score(y, np.r_[real_scores, fake_scores]))


def mean_auroc(real_scores, fake_scores, generator):  # AUROC per generator, then averaged over the test set
    return float(np.mean([auroc(real_scores, fake_scores[generator == g]) for g in sorted(set(generator))]))
