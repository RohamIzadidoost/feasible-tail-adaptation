"""Recompute manuscript results from cached scores; no torch or GPU required.

Run from the repository root: python audit_icassp.py
Original result CSVs are retained. Outputs go to manuscript_audit/.
Reconstructs the sorted manifests and seeded sampling used by protocol_a_public.
Checks the reconstructed labels against the historical AUC and accuracy before
using them. Array checksums and manifest checksums record the audited inputs.
"""
import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from metrics import compute_eer, threshold_diagnostics

OUT = Path("manuscript_audit")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifests():
    key_path = Path("data/dataset_1/DF-keys-full/keys/DF/CM/trial_metadata.txt")
    keys = {}
    for line in key_path.read_text().splitlines():
        p = line.split()
        if len(p) >= 8:
            keys[p[1]] = (int(p[5] == "spoof"), p[7], p[2])
    available = {Path(p).stem for p in glob.glob(
        "data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac", recursive=True)}
    rows = [(u, keys[u][0]) for u in sorted(available)
            if u in keys and keys[u][1] == "eval"]
    df = pd.DataFrame(rows, columns=["utt", "label"])
    itw = []
    for p in glob.glob("data/in_the_wild/**/*.wav", recursive=True):
        parts = Path(p).parts
        lab = 0 if "real" in parts else 1 if "fake" in parts else None
        if lab is not None:
            itw.append((str(Path(p).relative_to("data/in_the_wild")), lab))
    itw = pd.DataFrame(itw, columns=["utt", "label"]).sort_values("utt").reset_index(drop=True)
    phase = Counter(k[1] for k in keys.values())
    info = {"metadata_sha256": digest(key_path), "official_all_phases": len(keys),
            "phase_counts": dict(phase), "available_eval": len(df),
            "available_eval_fake": int(df.label.sum()),
            "available_eval_real": int((df.label == 0).sum()),
            "eval_coverage": len(df) / phase["eval"],
            "itw_n": len(itw), "itw_fake": int(itw.label.sum())}
    strata = []
    for phase_name in sorted(phase):
        for lab in [0, 1]:
            ids = {u for u, k in keys.items() if k[:2] == (lab, phase_name)}
            strata.append(dict(phase=phase_name, label=lab, official=len(ids),
                               available=len(ids & available), coverage=len(ids & available)/len(ids)))
    pd.DataFrame(strata).to_csv(OUT / "coverage.csv", index=False)
    for name, frame in [("df2021", df), ("itw", itw)]:
        info[name + "_manifest_sha256"] = hashlib.sha256(
            frame.to_csv(index=False).encode()).hexdigest()
    return {"df2021": df, "itw": itw}, info


def select_pool(ev, seed, sub, skew):
    ev = ev.copy()
    if skew:
        rng = np.random.RandomState(4242)
        f, r = ev.index[ev.label == 1].values, ev.index[ev.label == 0].values
        nr = int(round(len(f) * (1 - skew) / skew))
        if nr > len(r):
            nr = len(r)
            f = rng.choice(f, int(round(nr * skew / (1 - skew))), replace=False)
        keep = np.r_[f, rng.choice(r, nr, replace=False)]
        ev = ev.loc[np.sort(keep)].reset_index(drop=True)
    if sub:
        ev = ev.sample(min(sub, len(ev)), random_state=12345).sort_values("utt").reset_index(drop=True)
    adapt = np.random.RandomState(seed).choice(len(ev), min(20000, len(ev)), replace=False)
    return ev, np.sort(adapt)


def values(y, s, threshold=.5):
    pred = s >= threshold
    eer, _ = compute_eer(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1-tpr
    j = np.argmin(abs(fpr-fnr))
    d = threshold_diagnostics(y, s, threshold)
    unique, counts = np.unique(s, return_counts=True)
    return dict(n=len(y), prior=float(y.mean()), eer=100*eer,
                legacy_eer=50*(fpr[j]+fnr[j]), auc=roc_auc_score(y, s),
                accuracy=100*np.mean(pred == y),
                balanced_accuracy=50*(np.mean(pred[y == 1])+np.mean(~pred[y == 0])),
                fpr=100*np.mean(pred[y == 0]), fnr=100*np.mean(~pred[y == 1]),
                oracle_accuracy=100*d["oracle_accuracy"], threshold_gap=100*d["threshold_gap"],
                threshold=float(threshold), exact_one_pct=100*np.mean(s == 1),
                near_one_pct=100*np.mean(s >= 1-1e-6),
                largest_tie_pct=100*counts.max()/len(s), n_unique=len(unique))


def write_tables(scores, controls):
    d = scores[(scores.seed == 0) & (scores.eval_sub == 0)
               & (scores["skew"] == 0) & (scores.setting == "available_pool")]
    ck = "deepfense_w2v2_aasist_s2"
    rows = []
    for label, arm, q in [("Source", "source", .3),
                          ("Entropy (Tent-style)", "tent", .3),
                          ("ETA reliability ablation", "eta", .3),
                          ("SAR with scaled recovery", "sar", .3),
                          ("IM-PL (SHOT-inspired)", "shot", .3),
                          ("Fixed confidence 0.95", "ours_conf", .3),
                          (r"Symmetric $q=0.02$", "ours_fixed", .02),
                          (r"Symmetric $q=0.3$", "ours_fixed", .3),
                          ("BBSE tail budget", "ours_bbse", .3)]:
        r = d[(d.corpus == "df2021") & (d.ckpt == ck) & (d.arm == arm) & (d.q == q)].iloc[0]
        rows.append(label+f" & {r.eer:.2f} & {r.auc:.4f} & {r.accuracy:.2f} & {r.balanced_accuracy:.2f} & {r.threshold_gap:.2f}"+r" \\")
    c = controls[(controls.corpus == "df2021") & (controls.ckpt == ck)
                 & (controls.seed == 0) & (controls.eval_sub == 0)
                 & (controls["skew"] == 0) & (controls.q == .3)
                 & (controls.setting == "available_pool")]
    for label, rule in [("Source + median threshold", "median"), ("Source + BBSE quantile threshold", "bbse_quantile")]:
        r = c[c.rule == rule].iloc[0]
        rows.append(label+f" & {r.eer:.2f} & {r.auc:.4f} & {r.accuracy:.2f} & {r.balanced_accuracy:.2f} & {r.threshold_gap:.2f}"+r" \\")
    (OUT / "table_df_main.tex").write_text("\n".join(rows)+"\n")
    for corpus, ids in [("df2021", [("DF-2", ck), ("DF-42", "deepfense_w2v2_aasist_s42"),
                                     ("DF-240", "deepfense_w2v2_aasist_s240")]),
                        ("itw", [("DF-2", ck), ("DF-42", "deepfense_w2v2_aasist_s42"),
                                  ("DF-240", "deepfense_w2v2_aasist_s240"), ("WF", "ssl_aasist_wavefake")])]:
        rows = []
        for label, checkpoint in ids:
            line = label
            for arm in ["source", "ours_fixed", "ours_bbse"]:
                r = d[(d.corpus == corpus) & (d.ckpt == checkpoint) & (d.arm == arm) & (d.q == .3)].iloc[0]
                line += f" & {r.eer:.2f}/{r.accuracy:.2f}"
            rows.append(line+r" \\")
        (OUT / f"table_{corpus}_checkpoints.tex").write_text("\n".join(rows)+"\n")
    rows = []
    for skew in [.9, .95, .97]:
        p = scores[(scores.corpus == "itw") & np.isclose(scores["skew"], skew)
                   & (scores.setting == "available_pool")]
        line = f"{skew:.2f} & {int(p.iloc[0].n):,}"
        r = p[(p.arm == "source") & (p.q == .3)].iloc[0]
        line += f" & {r.accuracy:.2f}"
        for q in [.02, .1, .3]:
            r = p[(p.arm == "ours_fixed") & (p.q == q)].iloc[0]
            line += f" & {r.accuracy:.2f}"
        rows.append(line+r" \\")
    (OUT / "table_skew.tex").write_text("\n".join(rows)+"\n")


def main():
    OUT.mkdir(exist_ok=True)
    pools, info = manifests()
    print(json.dumps(info, indent=2), flush=True)
    results, controls, selections, provenance = [], [], [], []
    for corpus, folder, history_path in [
        ("df2021", "scores_protocol_a_public", "results_protocol_a_public.csv"),
        ("itw", "scores_protocol_a_public_itw", "results_protocol_a_public_itw.csv")]:
        hist = pd.read_csv(history_path)
        for col, default in [("seed", 0), ("eval_sub", 0), ("q", .3), ("skew_target", 0)]:
            if col not in hist:
                hist[col] = default
            hist[col] = hist[col].fillna(default)
        for path in sorted(Path(folder).glob("*.npy")):
            ckpt, arm, *tag = path.stem.split("__")
            if arm == "M":
                continue
            suffix = tag[0] if tag else ""
            m = re.fullmatch(r"s(\d+)_e(\d+)(?:_q([\d.]+))?(?:_k([\d.]+))?", suffix) if suffix else None
            if suffix and not m:
                raise ValueError(f"Unrecognized score suffix: {path}")
            seed, sub, q, skew = (int(m[1]), int(m[2]), float(m[3] or .3), float(m[4] or 0)) if m else (0, 0, .3, 0)
            ev, adapt = select_pool(pools[corpus], seed, sub, skew)
            y, s = ev.label.to_numpy(), np.load(path, allow_pickle=False)
            if s.shape != y.shape:
                raise ValueError(f"Manifest/score mismatch: {path}: {s.shape} vs {y.shape}")
            key = dict(corpus=corpus, ckpt=ckpt, arm=arm, seed=seed, eval_sub=sub, q=q, skew=skew)
            original = hist[(hist.ckpt == ckpt) & (hist.method == arm) & (hist.setting == "official_eval")
                            & (hist.seed == seed) & (hist.eval_sub == sub)
                            & np.isclose(hist.q, q) & np.isclose(hist.skew_target, skew)]
            # Some q-sweep runs reused the source row; q never changes the
            # frozen checkpoint or the evaluation manifest.
            if len(original) == 0 and arm == "source":
                original = hist[(hist.ckpt == ckpt) & (hist.method == arm)
                                & (hist.setting == "official_eval") & (hist.seed == seed)
                                & (hist.eval_sub == sub) & np.isclose(hist.skew_target, skew)]
                original = original.iloc[:1]
            v = values(y, s)
            if len(original) != 1:
                raise ValueError(f"Expected one historical row for {path}, got {len(original)}")
            row = original.iloc[0]
            if (abs(v["accuracy"] - row.acc) > .002 or abs(v["auc"] - row.auc) > .00011
                    or abs(v["legacy_eer"] - row.eer) > .002):
                raise ValueError(f"Historical metrics do not match reconstructed manifest: {path}")
            results.append(dict(**key, setting="available_pool", **v))
            mask = np.ones(len(y), dtype=bool)
            mask[adapt] = False
            if len(np.unique(y[mask])) == 2:
                results.append(dict(**key, setting="disjoint_eval", **values(y[mask], s[mask])))
            provenance.append(dict(file=str(path), sha256=digest(path), n=len(s)))
            if arm != "source":
                continue
            # Frozen-score, label-free threshold controls. BBSE uses only a
            # cached source confusion matrix and source predictions on adapt.
            matrix_path = Path("scores_protocol_a_public") / f"{ckpt}__M.npy"
            pi_hat = None
            if matrix_path.exists():
                matrix = np.load(matrix_path, allow_pickle=False)
                h = np.array([np.mean(s[adapt] < .5), np.mean(s[adapt] >= .5)])
                p = np.linalg.solve(matrix.T, h)
                p = np.clip(p, 1e-4, 1-1e-4)
                pi_hat = float(p[1]/p.sum())
            thresholds = {"median": float(np.median(s[adapt]))}
            if pi_hat is not None:
                thresholds["bbse_quantile"] = float(np.quantile(s[adapt], 1-pi_hat))
            for rule, threshold in thresholds.items():
                for setting, take in [("available_pool", np.ones(len(y), dtype=bool)), ("disjoint_eval", mask)]:
                    if len(np.unique(y[take])) == 2:
                        controls.append(dict(**key, rule=rule, setting=setting, pi_hat=pi_hat,
                                             **values(y[take], s[take], threshold)))
            if seed or sub or skew or q != .3:
                continue
            # Audit the initial pseudo-label masks, including all boundary ties
            # and the implementation's fake-tail overwrite on any overlap.
            for rule, lo, hi in [("symmetric", q, q)] + (
                [("bbse", 2*q*(1-pi_hat), 2*q*pi_hat)] if pi_hat is not None else []):
                a, ya = s[adapt], y[adapt]
                pl = np.full(len(a), -1)
                pl[a <= np.quantile(a, lo)] = 0
                pl[a >= np.quantile(a, 1-hi)] = 1
                for c, nominal in [(0, lo), (1, hi)]:
                    take = pl == c
                    actual = float(take.mean())
                    purity = float(np.mean(ya[take] == c)) if take.any() else np.nan
                    bound = min(1, np.mean(ya == c)/actual) if actual else np.nan
                    if take.any() and purity > bound+1e-12:
                        raise AssertionError("Counting bound violated")
                    selections.append(dict(**key, rule=rule, label=c, nominal=nominal,
                                           actual=actual, purity=purity, bound=bound))
    for name, rows in [("scores", results), ("threshold_controls", controls), ("initial_tails", selections)]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, float_format="%.10g")
    write_tables(pd.DataFrame(results), pd.DataFrame(controls))
    info["score_inputs"] = provenance
    info["method"] = "ROC interpolation with ties grouped; exhaustive distinct-score oracle; historical AUC/accuracy cross-check"
    import scipy
    import sklearn
    info["versions"] = {"numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    auxiliary = [Path("audit_icassp.py"), Path("metrics.py"),
                 Path("results_protocol_a_public.csv"), Path("results_protocol_a_public_itw.csv")]
    auxiliary += sorted(Path("scores_protocol_a_public").glob("*__M.npy"))
    info["auxiliary_inputs"] = [{"file": str(p), "sha256": digest(p)} for p in auxiliary]
    (OUT / "provenance.json").write_text(json.dumps(info, indent=2)+"\n")
    print(f"Verified {len(provenance)} score arrays; wrote {len(results)} metric rows to {OUT}")
    display = pd.DataFrame(results)
    display = display[(display.seed == 0) & (display.eval_sub == 0) & (display["skew"] == 0)
                      & (display.setting == "available_pool")]
    print(display[["corpus", "ckpt", "arm", "eer", "auc", "accuracy", "balanced_accuracy", "threshold_gap", "exact_one_pct", "near_one_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
