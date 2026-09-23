"""Regenerate every number in Tables 1 and 2 of the paper from the committed CSVs.

No GPU, no audio, no model weights: the tables are recomputed from cached score
arrays (already reduced to metrics in manuscript_audit/scores.csv) and from the
per-fold result files.

    python make_tables_icassp.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PI_FAKE_DF = 0.9722
DF_ARMS = [("source", "Source (as released)"), ("tent", "+ Tent"), ("eta", "+ ETA"),
           ("shot", "+ IM-PL"), ("sar", "+ SAR"),
           ("ours_fixed", "+ TTA, symmetric q=0.3"), ("ours_bbse", "+ FTA (ours)")]
PROTO_B_TARGETS = ["arabic", "asvspoof2019", "dataset2", "in_the_wild"]
PROTO_B_SEEDS = [0, 1, 2]


def table1():
    """Official ASVspoof2021-DF, mean over the three released checkpoints."""
    d = pd.read_csv("manuscript_audit/scores.csv")
    d = d[(d.corpus == "df2021") & (d.setting == "available_pool")
          & (d.seed == 0) & (d.eval_sub == 0) & (d.q == 0.3)]
    print("Table 1  official DF eval, mean over three checkpoints\n")
    print(f"{'':28s}{'EER%':>7}{'AUC':>7}{'acc':>8}{'bal.acc':>9}{'delta':>7}")
    for arm, label in DF_ARMS:
        g = d[d.arm == arm]
        if not len(g):
            continue
        print(f"{label:28s}{g.eer.mean():7.2f}{g.auc.mean():7.3f}"
              f"{g.accuracy.mean():8.2f}{g.balanced_accuracy.mean():9.2f}"
              f"{g.threshold_gap.mean():7.2f}")

    t = pd.read_csv("manuscript_audit/threshold_controls.csv")
    t = t[(t.corpus == "df2021") & (t.setting == "available_pool")
          & (t.eval_sub == 0) & (t.seed == 0)].drop_duplicates(["ckpt", "rule"])
    for rule, label in [("median", "+ median threshold"),
                        ("bbse_quantile", "+ BBSE threshold")]:
        g = t[t.rule == rule]
        print(f"{label:28s}{g.eer.mean():7.2f}{g.auc.mean():7.3f}"
              f"{g.accuracy.mean():8.2f}{g.balanced_accuracy.mean():9.2f}"
              f"{g.threshold_gap.mean():7.2f}")

    s2 = pd.read_csv("manuscript_audit/scores.csv")
    s2 = s2[(s2.corpus == "df2021") & (s2.setting == "available_pool")
            & (s2.ckpt.str.endswith("s2"))]
    extra = {"FTA without L_cons": s2[(s2.arm == "ours_bbse_nocons")],
             "fixed confidence 0.95": s2[(s2.arm == "ours_conf")],
             "estimator-free cap q=0.02": s2[(s2.arm == "ours_fixed") & (s2.q == 0.02)]}
    print("\n  single-checkpoint arms (seed 2):")
    for label, g in extra.items():
        if len(g):
            print(f"    {label:26s} EER {g.eer.iloc[0]:5.2f}   acc {g.accuracy.iloc[0]:6.2f}")


def df_nine_runs():
    """The 9/9, 8/9 and 7/9 counts quoted in Sec. 4.1."""
    d = pd.read_csv("manuscript_audit/scores.csv")
    d = d[(d.corpus == "df2021") & (d.q == 0.3)]
    print("\nSec. 4.1  nine runs (three checkpoints x three adaptation runs)")
    for setting in ("available_pool", "disjoint_eval"):
        p = d[d.setting == setting].pivot_table(
            index=["ckpt", "seed", "eval_sub"], columns="arm", values="eer")
        p = p.dropna(subset=["ours_bbse", "ours_fixed", "source"])
        print(f"  {setting:15s} n={len(p):2d}  FTA {p.ours_bbse.mean():5.2f}  "
              f"symmetric {p.ours_fixed.mean():5.2f}  source {p.source.mean():5.2f}  "
              f"| FTA<sym {(p.ours_bbse < p.ours_fixed).sum()}/{len(p)}  "
              f"FTA<src {(p.ours_bbse < p.source).sum()}/{len(p)}")

    t = pd.read_csv("manuscript_audit/initial_tails.csv")
    t = t[(t.corpus == "df2021") & (t.label == 0)]
    print("  pseudo-real tail purity: symmetric "
          f"{t[t.rule == 'symmetric'].purity.mean():.3f} -> FTA "
          f"{t[t.rule == 'bbse'].purity.mean():.3f}")


def table2():
    """Protocol B: twelve leave-one-corpus-out folds."""
    d = pd.read_csv("results_adaptive.csv")
    d = d[d.target.isin(PROTO_B_TARGETS) & d.seed.isin(PROTO_B_SEEDS)
          & (d.setting == "transductive")]
    key = ["seed", "target"]
    src = d[d.method == "source"].drop_duplicates(key).set_index(key).sort_index()

    def row(label, frame):
        frame = frame.drop_duplicates(key).set_index(key).sort_index()
        if len(frame) != len(src):
            return
        wins = int((frame.eer.values < src.eer.values).sum())
        try:
            p = wilcoxon(frame.eer.values, src.eer.values).pvalue
        except ValueError:
            p = float("nan")
        print(f"{label:26s}{frame.eer.mean():7.2f} +-{frame.eer.std():5.2f}"
              f"{frame.acc.mean():8.2f}   {wins:2d}/{len(frame)} (p={p:.3f})")

    print("\nTable 2  Protocol B, twelve folds (four targets x three seeds)\n")
    print(f"{'':26s}{'EER%':>7}   {'s.d.':>4}{'acc':>8}   >src (p)")
    row("Source", d[d.method == "source"])
    for m, label in [("tent", "+ Tent"), ("eta", "+ ETA"), ("eata", "+ EATA"),
                     ("sar", "+ SAR"), ("shot", "+ IM-PL"),
                     ("ours_E4", "+ FTA (ours)")]:
        row(label, d[d.method == m])
    e = pd.read_csv("results_ext.csv")
    e = e[(e.setting == "transductive") & (e.family == "xlsr")
          & e.seed.isin(PROTO_B_SEEDS) & e.target.isin(PROTO_B_TARGETS)]
    row("+ AdaBN", e[e.method == "bn_only"])
    row("Target-supervised oracle", e[e.method == "oracle"])
    for f, label in [("results_dann.csv", "DANN"), ("results_asdg.csv", "ASDG-style")]:
        a = pd.read_csv(f)
        row(label, a[a.target.isin(PROTO_B_TARGETS) & a.seed.isin(PROTO_B_SEEDS)])


def breadth():
    """The 369-cell third-party arm quoted in Sec. 4.7."""
    r = pd.read_csv("results_public_ckpt_multi.csv")
    t = r[r.setting == "transductive"]
    key = ["family", "target", "seed"]
    s = t[t.method == "source"].groupby(key).mean(numeric_only=True)
    o = t[t.method == "ours"].groupby(key).mean(numeric_only=True)
    m = s.join(o, lsuffix="_s", rsuffix="_o", how="inner")
    ds, do = 100 - m.eer_s - m.acc_s, 100 - m.eer_o - m.acc_o
    print(f"\nSec. 4.7  {len(m)} third-party cells over "
          f"{m.reset_index()[['family', 'target']].drop_duplicates().shape[0]} pairs: "
          f"deficit {ds.mean():.2f} -> {do.mean():.2f}, improved {(do < ds).sum()}/{len(m)}, "
          f"p={wilcoxon(ds, do).pvalue:.1e}")


if __name__ == "__main__":
    table1()
    df_nine_runs()
    table2()
    breadth()
