"""Score-resolution diagnostics for every cached arm.

Near-ceiling scores and exact ties are different: the historical Tent run has
98.07% of scores >= 1-1e-6, but its largest exact tie is 93.02%. AUC and EER
summarize different parts of the ROC and can move in opposite directions.
Report both, group exact ties, and never describe near-ceiling mass as an exact
tie. metrics.compute_eer now interpolates the ROC crossing.

    python score_resolution.py [scores_dir] [--corpus df2021]
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir", nargs="?", default="scores_protocol_a_public")
    ap.add_argument("--corpus", default="df2021")
    a = ap.parse_args()
    os.environ["PUBA_CORPUS"] = a.corpus
    import protocol_a_public as PA
    from sklearn.metrics import roc_auc_score
    from metrics import compute_eer

    ev = PA.build_eval()
    y = ev.label.values.astype(int)
    rows = []
    for f in sorted(os.listdir(a.scores_dir)):
        if not f.endswith(".npy") or "__M" in f:
            continue
        s = np.load(os.path.join(a.scores_dir, f))
        if len(s) != len(y):
            continue
        ckpt, arm = f[:-4].split("__", 1)
        eer, _ = compute_eer(y, s)
        # ties at the numerical ceiling/floor are where entropy minimisation
        # hides: they cost AUC but can leave EER looking healthy
        rows.append(dict(
            ckpt=ckpt, arm=arm, n=len(s),
            eer=round(eer * 100, 3), auc=round(float(roc_auc_score(y, s)), 4),
            n_unique=int(len(np.unique(s))),
            pct_at_max=round(float(np.mean(s >= 1 - 1e-6)) * 100, 2),
            pct_at_min=round(float(np.mean(s <= 1e-6)) * 100, 2),
            pct_exact_one=round(float(np.mean(s == 1.0)) * 100, 2),
            pct_exact_zero=round(float(np.mean(s == 0.0)) * 100, 2),
            largest_tie_pct=round(float(pd.Series(s).value_counts().iloc[0]
                                        / len(s)) * 100, 2)))
    df = pd.DataFrame(rows).sort_values(["ckpt", "arm"])
    out = f"score_resolution_{a.corpus}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")
    bad = df[df.largest_tie_pct > 50]
    if len(bad):
        print("\nArms with >50% a single exact score. Report tie-aware ROC "
              "metrics and distinguish exact from near-ceiling saturation:")
        print(bad[["ckpt", "arm", "eer", "auc", "largest_tie_pct", "n_unique"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
