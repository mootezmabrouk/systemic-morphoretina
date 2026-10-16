"""Evaluation metrics (CdCF section 4.1).

Place at: scripts/evaluate.py

Image level : sensitivity, specificity, precision, F1 at a chosen grade
              threshold, plus the 5x5 confusion matrix.
Lesion level: per-type F1, pixel-wise and object-wise.

CLI (grade evaluation from a CSV with columns: image,true_grade,pred_grade):
    python scripts/evaluate.py results.csv --threshold 1
A pred_grade of "NE" (not evaluable, grades 3-4) is counted separately.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage as ndi


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def binary_metrics(y_true, y_pred) -> dict:
    """Metrics for boolean arrays (True = positive)."""
    t = np.asarray(y_true, bool)
    p = np.asarray(y_pred, bool)
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    tn = int((~t & ~p).sum())
    fn = int((t & ~p).sum())
    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * prec * sens, prec + sens)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, sensitivity=sens,
                specificity=spec, precision=prec, f1=f1)


def grade_binary_metrics(true_grades, pred_grades, threshold: int = 1) -> dict:
    """Positive = grade >= threshold (1: any DR, 2: referable)."""
    t = np.asarray(true_grades) >= threshold
    p = np.asarray(pred_grades) >= threshold
    return binary_metrics(t, p)


def confusion_matrix_5x5(true_grades, pred_grades) -> np.ndarray:
    """Rows = true grade 0..4, columns = predicted grade 0..4."""
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(true_grades, pred_grades):
        cm[int(t), int(p)] += 1
    return cm


def pixel_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    return binary_metrics(gt.ravel(), pred.ravel())


def lesion_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Object-wise: a GT lesion is detected if any predicted object overlaps
    it; a predicted object is correct if it overlaps any GT lesion."""
    gt_lab, n_gt = ndi.label(gt)
    pr_lab, n_pr = ndi.label(pred)
    gt_hit = len(np.unique(gt_lab[(pr_lab > 0) & (gt_lab > 0)]))
    pr_hit = len(np.unique(pr_lab[(pr_lab > 0) & (gt_lab > 0)]))
    rec = _div(gt_hit, n_gt)
    prec = _div(pr_hit, n_pr)
    f1 = _div(2 * prec * rec, prec + rec)
    return dict(n_gt=n_gt, n_pred=n_pr, recall=rec, precision=prec, f1=f1)


def _main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--threshold", type=int, default=1)
    a = ap.parse_args()

    df = pd.read_csv(a.csv, dtype={"pred_grade": str})
    ne = df["pred_grade"].str.upper() == "NE"
    print(f"Not evaluable (excluded): {int(ne.sum())} of {len(df)}")
    df = df[~ne]
    t = df["true_grade"].astype(int).to_numpy()
    p = df["pred_grade"].astype(int).to_numpy()

    m = grade_binary_metrics(t, p, a.threshold)
    print(f"Threshold: grade >= {a.threshold}")
    for k, v in m.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Confusion matrix (rows true, cols predicted):")
    print(confusion_matrix_5x5(t, p))


if __name__ == "__main__":
    _main()