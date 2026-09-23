"""
Evaluation protocols for DA-RAD.

Protocol A (official, comparability): pooled EER on the ASVspoof2021-DF eval set,
scored against the official CM keys (trial_metadata.txt) we already hold under
data/dataset_1/DF-keys-full/.../CM/. This is the number comparable to the
literature (wav2vec2 ~5%, AASIST ~3%).

Protocol B (generalisation): a leave-one-corpus-out matrix. Given per-(train,test)
corpus EERs, we report the in-domain diagonal, the average and worst-case
off-diagonal (cross-corpus) EER, and the generalisation gap.

Scores follow the project convention: score = P(fake), label 1 = fake / spoof.
EER itself is reused from metrics.compute_eer.
"""

import glob
import os

import numpy as np
import pandas as pd

from metrics import compute_eer

DF_KEYS_DEFAULT = "data/dataset_1/DF-keys-full/keys/DF/CM/trial_metadata.txt"


# --------------------------------------------------------------------------
# Protocol A — official ASVspoof2021 DF
# --------------------------------------------------------------------------
def load_df_keys(meta_path: str = DF_KEYS_DEFAULT) -> dict:
    """Return {utt_id: (1 if spoof else 0, phase)} from the official DF CM keys.

    Line format: <speaker> <utt_id> <codec> <source> <attack> <label> <trim> <phase> ...
    (label at column index 5 is 'bonafide' or 'spoof'; phase at index 7 is
    'eval', 'progress', or 'hidden'.)
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"DF key file not found: {meta_path}")
    keys = {}
    with open(meta_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            keys[p[1]] = (1 if p[5] == "spoof" else 0, p[7])
    return keys


def score_official_df(scores: dict, meta_path: str = DF_KEYS_DEFAULT,
                      phase: str | None = "eval") -> dict:
    """Pooled EER over the DF eval trials.

    Args:
        scores: {utt_id: P(fake)} produced by a model over the DF eval set.
        meta_path: official CM keys.
        phase: restrict to one partition of the key file. Defaults to 'eval',
            the partition the literature reports; the 'progress' subset was the
            live challenge leaderboard and 'hidden' is a held-back subset, so
            pooling all three does NOT give a comparable number. Pass None to
            pool every partition deliberately.
    Returns dict with eer (%), threshold, n_scored, n_missing, phase.
    """
    keys = load_df_keys(meta_path)
    y, s, missing = [], [], 0
    for utt, (label, utt_phase) in keys.items():
        if phase is not None and utt_phase != phase:
            continue
        if utt in scores:
            y.append(label)
            s.append(scores[utt])
        else:
            missing += 1
    if len(set(y)) < 2:
        raise ValueError("need both bonafide and spoof trials to compute EER")
    eer, thr = compute_eer(np.array(y), np.array(s))
    return {"eer": eer * 100, "threshold": thr, "n_scored": len(y),
            "n_missing": missing, "phase": phase or "all"}


# --------------------------------------------------------------------------
# Protocol B — cross-corpus matrix
# --------------------------------------------------------------------------
def cross_corpus_summary(matrix: pd.DataFrame) -> dict:
    """Given a square EER matrix (rows=train corpus, cols=test corpus, %),
    return in-domain (diagonal) mean, cross-corpus mean/worst, and the gap.
    """
    m = matrix.to_numpy(dtype=float)
    diag = np.diag(m)
    off = m[~np.eye(m.shape[0], dtype=bool)]
    in_domain = float(np.nanmean(diag))
    cross_mean = float(np.nanmean(off))
    cross_worst = float(np.nanmax(off))
    return {
        "in_domain_eer": in_domain,
        "cross_corpus_eer_mean": cross_mean,
        "cross_corpus_eer_worst": cross_worst,
        "generalization_gap": cross_mean - in_domain,
    }


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # (1) EER scoring logic on synthetic scores
    rng = np.random.default_rng(0)
    fake_keys = {f"u{i}": (i % 2) for i in range(2000)}          # alternate labels
    # scores correlated with label -> low-ish EER
    scores = {u: float(np.clip(0.5 * lab + 0.25 * rng.standard_normal() + 0.25, 0, 1))
              for u, lab in fake_keys.items()}
    # write a tiny fake key file and score it
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for u, lab in fake_keys.items():
            # 8 columns: phase must be present at index 7 for load_df_keys
            fh.write(f"spk {u} codec src attack "
                     f"{'spoof' if lab else 'bonafide'} notrim eval\n")
        tmp = fh.name
    out = score_official_df(scores, tmp)
    os.unlink(tmp)
    print(f"synthetic official-DF scoring: EER {out['eer']:.2f}%  "
          f"n={out['n_scored']}  missing={out['n_missing']}")
    assert 0 <= out["eer"] <= 50

    # (2) cross-corpus summary on a toy matrix
    corpora = ["ASV", "ITW", "ds2"]
    toy = pd.DataFrame([[3.0, 25.0, 30.0],
                        [22.0, 4.0, 28.0],
                        [26.0, 24.0, 5.0]], index=corpora, columns=corpora)
    summ = cross_corpus_summary(toy)
    print("cross-corpus summary:", {k: round(v, 2) for k, v in summ.items()})
    assert summ["generalization_gap"] > 0

    # (3) if the real DF keys are present, parse & report counts (no scoring)
    if os.path.exists(DF_KEYS_DEFAULT):
        keys = load_df_keys()
        n_spoof = sum(lab for lab, _ in keys.values())
        from collections import Counter
        phases = Counter(ph for _, ph in keys.values())
        print(f"real DF keys parsed: {len(keys)} trials "
              f"({n_spoof} spoof / {len(keys) - n_spoof} bonafide)")
        print(f"  phase partitions: {dict(phases)}")
    else:
        print(f"(real DF keys not found at {DF_KEYS_DEFAULT}; skipped)")
    print("eval_protocol.py self-test passed.")
