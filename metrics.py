"""
Evaluation metrics for the deepfake-audio classifier.

The audio-anti-spoofing field reports Equal Error Rate (EER), not raw accuracy —
accuracy hides the "predict everything fake" failure mode (the paper's LR/KNC hit
0.00 recall on real yet still reported 84–93% "accuracy"). This module gives EER
alongside accuracy, balanced accuracy, ROC-AUC, and a per-class report so that
failure mode can't hide.

Convention: label 1 = fake (the positive/spoof class), label 0 = real, matching
LABEL_TO_IDX in deepfake_dataset.py. `scores` are the model's predicted P(fake).

Usage:
    from metrics import compute_eer, evaluation_report
    report = evaluation_report(y_true, p_fake)   # prints + returns a dict
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(labels, scores):
    """EER at the linearly interpolated crossing of the empirical ROC.

    Equal scores are grouped by ``roc_curve`` rather than ordered arbitrarily.
    Returns (eer, threshold). The threshold is the lower endpoint of the
    crossing segment: with discrete scores no deterministic threshold need
    attain the interpolated EER. Labels must contain both 0 and 1.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("labels and scores must be nonempty, matching 1-D arrays")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("EER requires both binary classes, encoded as 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1,
                                    drop_intermediate=False)
    fnr = 1 - tpr
    delta = fpr - fnr
    idx = int(np.flatnonzero(delta >= 0)[0])
    if delta[idx] == 0:
        return float(fpr[idx]), float(thresholds[idx])
    weight = -delta[idx - 1] / (delta[idx] - delta[idx - 1])
    eer = float(fpr[idx - 1] + weight * (fpr[idx] - fpr[idx - 1]))
    return eer, float(thresholds[idx])


def threshold_diagnostics(labels, scores, threshold=0.5):
    """Exact best-threshold accuracy and gap, for labelled diagnostics only.

    Includes the all-negative and all-positive decisions. Ties move together;
    no rounding, candidate subsampling, or label-dependent tie-breaking is used.
    The oracle threshold is selected on these same labels and is NOT a
    deployable label-free control or a held-out performance estimate.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    # Reuse the validation above.
    compute_eer(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1,
                                    drop_intermediate=False)
    prior = float(labels.mean())
    accs = prior * tpr + (1 - prior) * (1 - fpr)
    best = int(np.argmax(accs))
    fixed = float(np.mean((scores >= threshold) == labels))
    return {"oracle_accuracy": float(accs[best]),
            "oracle_threshold": float(thresholds[best]),
            "threshold_gap": max(0.0, float(accs[best]) - fixed)}


def evaluation_report(labels, scores, threshold=0.5, verbose=True):
    """Full metric bundle from ground-truth labels and P(fake) scores.

    Returns a dict with accuracy, balanced_accuracy, eer, eer_threshold, auc.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)

    eer, eer_thr = compute_eer(labels, scores)
    result = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1_fake": float(f1_score(labels, preds, pos_label=1)),
        "eer": eer,
        "eer_threshold": eer_thr,
        "auc": float(roc_auc_score(labels, scores)),
    }

    if verbose:
        print(f"EER               : {result['eer'] * 100:.2f}%  (lower is better)")
        print(f"ROC-AUC           : {result['auc']:.4f}")
        print(f"Accuracy @{threshold:.2f}    : {result['accuracy'] * 100:.2f}%")
        print(f"Balanced accuracy : {result['balanced_accuracy'] * 100:.2f}%")
        print("\nPer-class report (0=real, 1=fake):")
        print(classification_report(labels, preds, target_names=["real", "fake"], digits=4))

    return result


if __name__ == "__main__":
    # quick self-test on synthetic scores
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    # scores correlated with labels + noise
    s = np.clip(0.5 * y + 0.25 * rng.standard_normal(2000) + 0.25, 0, 1)
    evaluation_report(y, s)
