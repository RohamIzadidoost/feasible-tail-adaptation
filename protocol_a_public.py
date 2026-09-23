"""Protocol A on PUBLISHED checkpoints: the official ASVspoof2021-DF eval.

Why this exists (ICASSP_REVIEW_AND_PLAN.md, W2): every EER gain in the paper is
measured on our own source model, which scores 26.33% on the one protocol with
published baselines (Wav2Vec2-AASIST 8.54%, AASIST 2.87%, challenge top-1
~15.6%). A reviewer reads "relative gain on a weak model" and discounts the
whole diagnosis as an artefact of under-training.

This script removes that objection by running the method on somebody else's
SOTA-grade weights, on the official protocol, at the protocol's real 97%-spoof
class balance:

  * checkpoints: DeepFense ASV19_Wav2Vec2_AASIST_NoAug_Seed{2,42,240} -- XLS-R
    300M + AASIST trained on ASVspoof2019-LA train, i.e. exactly the Protocol-A
    training condition the published baselines use. Optionally ash56/ssl-aasist
    (WaveFake-trained), which is a different training condition and is reported
    separately.
  * arms: source / naive fixed-q TTA / prior-corrected (BBSE) TTA /
    label-free median-threshold control.
  * eval: official DF CM keys, phase='eval', pooled EER (eval_protocol.py).

Labels are used ONLY for scoring, never in adaptation. The BBSE prior estimate
reads the ASVspoof2019-LA *train* split, which is the checkpoint's own labelled
training data and therefore legitimately available at deployment.

    PUBA_SMOKE=1 python protocol_a_public.py          # ~10 min sanity run
    python protocol_a_public.py                       # full
"""

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

import public_ckpt_tta as P
import tta_baselines as TB
from eval_protocol import DF_KEYS_DEFAULT, load_df_keys, score_official_df
from metrics import compute_eer, threshold_diagnostics

SMOKE = os.environ.get("PUBA_SMOKE", "0") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SR = 16000
BATCH = int(os.environ.get("PUBA_BATCH", "32"))
DECODE_WORKERS = 16

# Q is the symmetric tail budget. The published value is 0.3, which is valid
# only while q <= min(pi, 1-pi): under a good ranking the bottom-q bucket can
# hold at most the pool's real clips, so its purity is min(1, pi_real/q) --
# verified against labels at r=0.9989, mean abs error 0.019. ITW satisfies it
# (min(.372,.628)=.372 > .3, purity .94-.99); DF violates it badly
# (min(.972,.028)=.028 << .3, purity .086) and that single inequality is why the
# published configuration destroys a checkpoint there. PUBA_Q sweeps it.
Q = float(os.environ.get("PUBA_Q", "0.3"))
LAMBDA_CONS, TTA_LR = 0.3, 1e-4
TTA_EPOCHS = int(os.environ.get("PUBA_EPOCHS", "4"))

DF_PARTS = "data/dataset_1/ASVspoof2021_DF_eval_part0*/**/*.flac"
LA19_PROTO = "data/asvspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
LA19_FLAC = "data/asvspoof2019_LA/ASVspoof2019_LA_train/flac"

# PUBA_CORPUS selects the evaluation corpus:
#   df2021 -- the official ASVspoof2021-DF eval partition (97% spoof, the
#             protocol with published baselines)
#   itw    -- the complete In-the-Wild benchmark, all 31,779 clips (62.8% spoof)
# Both are run whole, not subsampled: the paper's existing third-party arm caps
# every pool at 6,000 clips, and a reviewer is entitled to the standard
# benchmark at its published size.
CORPUS = os.environ.get("PUBA_CORPUS", "df2021")

# PUBA_SEED varies the adaptation RNG and the unlabelled adapt-pool draw, so the
# headline can be given an error bar. PUBA_EVAL_SUB scores a fixed random subset
# of the eval instead of all of it -- the SAME subset for every seed and arm, so
# the comparison stays paired. Seed 0 with no subsample is the official-protocol
# number; the subsampled seeds carry the variance.
SEED = int(os.environ.get("PUBA_SEED", "0"))

# PUBA_SKEW resamples the evaluation pool to a target P(fake), holding the audio
# and the checkpoint fixed. This is the controlled test of Eq. (3): the validity
# condition q <= min(pi, 1-pi) is a claim about prevalence, so the way to test it
# is to *intervene* on prevalence rather than to compare corpora that differ in
# a dozen other ways. The fake class is kept whole and the real class thinned,
# so the pool shrinks as skew rises but the spoof audio is identical throughout.
SKEW = float(os.environ.get("PUBA_SKEW", "0")) or None
EVAL_SUB = int(os.environ.get("PUBA_EVAL_SUB", "0")) or None

# Where each checkpoint's own labelled training data lives. BBSE needs the
# source confusion matrix M, which is measured on data the deployer legitimately
# has: the checkpoint's training corpus. A checkpoint whose training data is
# undocumented cannot get an honest M and is therefore excluded from the BBSE
# arm rather than fed a proxy.
BBSE_SOURCE = {
    "deepfense_w2v2_aasist_s2": "la19",
    "deepfense_w2v2_aasist_s42": "la19",
    "deepfense_w2v2_aasist_s240": "la19",
    "ssl_aasist_wavefake": "wavefake",
}

CKPTS = os.environ.get("PUBA_CKPTS",
                       "deepfense_w2v2_aasist_s2,deepfense_w2v2_aasist_s42,"
                       "deepfense_w2v2_aasist_s240").split(",")
ARMS = os.environ.get("PUBA_ARMS", "source,ours_fixed,ours_bbse").split(",")

if SMOKE:
    ADAPT_N = int(os.environ.get("PUBA_ADAPT_N", "400"))
    MAX_EVAL = int(os.environ.get("PUBA_MAX_EVAL", "2000"))
    BBSE_N = int(os.environ.get("PUBA_BBSE_N", "200"))
    TTA_EPOCHS = 1
    CKPTS = CKPTS[:1]
else:
    ADAPT_N = int(os.environ.get("PUBA_ADAPT_N", "20000"))
    MAX_EVAL, BBSE_N = None, 4000

SUFFIX = ("_smoke" if SMOKE else "") + ("" if CORPUS == "df2021" else f"_{CORPUS}")
RESULTS_CSV = f"results_protocol_a_public{SUFFIX}.csv"
SCORES_DIR = f"scores_protocol_a_public{SUFFIX}"
LOG_FILE = f"run_log_protocol_a_public{SUFFIX}.txt"
os.makedirs(SCORES_DIR, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def record(**row):
    """Append one result row, reconciling the schema rather than appending blind.

    Arms carry different extra columns (the BBSE arm reports its prior estimate
    and tail budget; the source arm has none), so a plain `mode="a"` append
    writes rows with more fields than the header the first row created --
    producing a CSV that pandas cannot parse and, worse, one where a careless
    reader silently mis-assigns values to columns. Rewriting the union schema
    costs nothing at these row counts and makes the file always well-formed.
    """
    new = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        try:
            old = pd.read_csv(RESULTS_CSV)
        except Exception:
            old = _read_ragged(RESULTS_CSV)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(RESULTS_CSV, index=False)


# Extra columns each arm appends, in the order `main()` sets them. Used only to
# repair a file written by the earlier append-blind version.
_RAGGED_EXTRA = {
    "ours_fixed": ["adapt_min"],
    "ours_bbse": ["pi_fake_hat", "pi_fake_true", "lo_frac", "hi_frac", "adapt_min"],
}


def _read_ragged(path):
    """Parse a results file written before record() reconciled schemas."""
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    base = lines[0].split(",")
    rows = []
    for l in lines[1:]:
        parts = l.split(",")
        d = dict(zip(base, parts[:len(base)]))
        for k, v in zip(_RAGGED_EXTRA.get(d.get("method", ""), []), parts[len(base):]):
            d[k] = v
        rows.append(d)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c in ("ckpt", "method", "setting", "smoke"):
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().any():
            df[c] = conv
    return df


def done_rows():
    if not os.path.exists(RESULTS_CSV):
        return set()
    try:
        d = pd.read_csv(RESULTS_CSV)
    except Exception:
        d = _read_ragged(RESULTS_CSV)
    if "seed" not in d:
        d = d.assign(seed=0)
    if "eval_sub" not in d:
        d = d.assign(eval_sub=0)
    if "q" not in d:
        d = d.assign(q=0.3)
    # NB: a column literally named `skew` shadows DataFrame.skew(), so `d.skew`
    # returns the method and `.fillna` on it raises. Named `skew_target` and
    # accessed with brackets throughout.
    if "skew_target" not in d.columns:
        d = d.assign(skew_target=0.0)
    return set(zip(d["ckpt"], d["method"], d["setting"],
                   d["seed"].fillna(0).astype(int),
                   d["eval_sub"].fillna(0).astype(int),
                   d["q"].fillna(0.3).round(3),
                   d["skew_target"].fillna(0.0).round(4)))


def _key(ckpt, method, setting):
    return (ckpt, method, setting, SEED, EVAL_SUB or 0, round(Q, 3),
            round(SKEW or 0.0, 4))


# ------------------------------------------------------------------ manifests
def build_df_eval():
    import glob
    keys = load_df_keys(DF_KEYS_DEFAULT)
    rows = []
    for p in glob.glob(DF_PARTS, recursive=True):
        utt = os.path.basename(p)[:-5]
        k = keys.get(utt)
        if k is not None and k[1] == "eval":
            rows.append((p, k[0], utt))
    df = pd.DataFrame(rows, columns=["path", "label", "utt"])
    return df.sort_values("utt").reset_index(drop=True)


def build_la19_train():
    rows = []
    for line in open(LA19_PROTO):
        p = line.split()
        if len(p) >= 5:
            rows.append((f"{LA19_FLAC}/{p[1]}.flac", 1 if p[-1] == "spoof" else 0))
    return pd.DataFrame(rows, columns=["path", "label"])


def build_wavefake():
    import glob
    rows = [(w, 0 if "/real/" in w else 1)
            for w in glob.glob("data/hf_wavefake/*/*.wav")]
    return pd.DataFrame(rows, columns=["path", "label"])


def build_itw():
    """The complete In-the-Wild benchmark: 31,779 clips, label from the path."""
    import glob
    rows = []
    for w in glob.glob("data/in_the_wild/**/*.wav", recursive=True):
        parts = w.split(os.sep)
        lab = 0 if "real" in parts else 1 if "fake" in parts else None
        if lab is not None:
            rows.append((w, lab, os.path.relpath(w, "data/in_the_wild")))
    df = pd.DataFrame(rows, columns=["path", "label", "utt"])
    return df.sort_values("utt").reset_index(drop=True)


def build_eval():
    if CORPUS == "df2021":
        return build_df_eval()
    if CORPUS == "itw":
        return build_itw()
    raise ValueError(f"unknown PUBA_CORPUS={CORPUS!r}")


def build_bbse_source(kind):
    if kind == "la19":
        df = build_la19_train()
    elif kind == "wavefake":
        df = build_wavefake()
    else:
        return None
    df = df[df.path.map(os.path.exists)].reset_index(drop=True)
    return df if len(df) else None


# ------------------------------------------------------------------ audio
def decode_batch(paths, crop):
    buf = torch.empty((len(paths), crop), dtype=torch.float16)
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as ex:
        for i, w in enumerate(ex.map(lambda p: P.load_clip(p, crop), paths)):
            buf[i] = torch.from_numpy(np.ascontiguousarray(w))
    return buf


@torch.no_grad()
def stream_score(model, paths, crop, fake_col, chunk=2048, tag=""):
    """Score a large manifest without caching it whole.

    Decode and GPU work overlap: measured in isolation this box decodes 533
    clips/s and scores 204 clips/s, so running them in series caps throughput at
    147 clips/s. One prefetch thread hides the decode entirely and the full
    400k-clip official eval drops from ~45 min to ~33 min per pass -- times the
    nine passes this study needs, that is four GPU-hours saved.
    """
    model.eval()
    out = np.zeros(len(paths), dtype=np.float32)
    t0 = time.time()
    starts = list(range(0, len(paths), chunk))
    pre = ThreadPoolExecutor(max_workers=1)
    nxt = pre.submit(decode_batch, paths[starts[0]:starts[0] + chunk], crop)
    for k, c0 in enumerate(starts):
        sl = slice(c0, min(c0 + chunk, len(paths)))
        buf = nxt.result()
        if k + 1 < len(starts):
            n0 = starts[k + 1]
            nxt = pre.submit(decode_batch, paths[n0:n0 + chunk], crop)
        got = []
        for i in range(0, len(buf), BATCH):
            x = buf[i:i + BATCH].to(DEVICE, non_blocking=True).float()
            with P.amp_ctx(DEVICE):
                logits = model(x)[0]
            got.append(torch.softmax(logits.float(), 1)[:, fake_col].cpu())
        out[sl] = torch.cat(got).numpy()
        del buf
        done = min(c0 + chunk, len(paths))
        if c0 == 0 or done % (chunk * 20) == 0 or done == len(paths):
            rate = done / max(time.time() - t0, 1e-9)
            log(f"    {tag}{done}/{len(paths)}  {rate:.0f} clips/s  "
                f"eta {(len(paths) - done) / max(rate, 1e-9) / 60:.1f} min")
    pre.shutdown(wait=True)
    return out


# ------------------------------------------------------------------ BBSE
def bbse_pi_fake(model, crop, fake_col, la19, target_scores, seed=0):
    """P(fake) on the target pool by black-box shift estimation.

    M[y, yhat] = P_source(yhat | y) measured on a class-balanced slice of the
    checkpoint's OWN labelled training corpus (ASVspoof2019-LA train); q is the
    target prediction histogram. p = M^{-T} q. Reads source error structure and
    target prediction rate only -- never the shape of the target score
    distribution, which is what miscalibration corrupts.
    """
    bal = pd.concat([g.sample(min(len(g), BBSE_N // 2), random_state=seed)
                     for _, g in la19.groupby("label")]).reset_index(drop=True)
    s_src = stream_score(model, bal.path.tolist(), crop, fake_col, tag="bbse-M ")
    yh = (s_src >= 0.5).astype(int)
    y = bal.label.values.astype(int)
    M = np.array([[np.mean(yh[y == 0] == 0), np.mean(yh[y == 0] == 1)],
                  [np.mean(yh[y == 1] == 0), np.mean(yh[y == 1] == 1)]])
    q = np.array([np.mean((target_scores >= 0.5) == 0),
                  np.mean((target_scores >= 0.5) == 1)])
    try:
        p = np.linalg.solve(M.T, q)
    except np.linalg.LinAlgError:
        return 0.5, M, q
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(p[1] / p.sum()), M, q


def gmm_pi_fake(scores, seed=0):
    """Target prior from a 2-component Gaussian mixture on the score histogram.

    Reads the SHAPE of the target score distribution rather than a thresholded
    count. That distinction turned out to matter: on In-the-Wild every
    prediction-rate estimator (BBSE, SLD, mean-probability) is off by ~0.23
    because the checkpoint predicts 58.9% fake on a 37.2%-fake pool -- which is
    just its +16.7-point calibration deficit passing straight through -- while
    this one is off by 0.05. A thresholded count inherits the broken
    calibration; a histogram does not.
    """
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, random_state=seed, n_init=3).fit(
        np.asarray(scores).reshape(-1, 1))
    return float(g.weights_[int(np.argmax(g.means_.flatten()))])


def tail_budget(q_base, pi_real):
    """Split a total confident budget 2*q_base by the estimated prior.

    Same rule as adaptive_tta.tail_budget: the pseudo-real bucket may not be
    larger than the estimated real fraction, so a 97%-spoof pool stops scooping
    up 30% of the pool as 'real'.
    """
    import adaptive_tta as AT
    return AT.tail_budget(q_base, pi_real)


# ------------------------------------------------------------------ adaptation
CONF_TAU = float(os.environ.get("PUBA_CONF_TAU", "0.95"))


def adapt(model, buf, crop, fake_col, lo_frac, hi_frac, epochs=TTA_EPOCHS,
          use_cons=True, conf_thresh=False, tag=""):
    """Confident-tail pseudo-label self-training + channel consistency.

    lo_frac/hi_frac are the pseudo-real / pseudo-fake budgets. (Q, Q) reproduces
    the published fixed-q method exactly.

    `conf_thresh=True` replaces the quantile rule with a fixed CONFIDENCE
    threshold (p >= CONF_TAU -> fake, p <= 1-CONF_TAU -> real), which is the
    obvious prior-free alternative and the first thing a reader will propose.
    The two rules fail on opposite axes, which is the point of running both:
    a quantile rule is calibration-free but assumes the prior; a confidence
    rule is prior-free but assumes calibration -- and calibration is precisely
    the property this paper's measurements say does NOT transfer across corpora.
    """
    trainable, info = P.set_tta_params(model)
    n_bn = P.freeze_batchnorm(model)
    rule = (f"confidence threshold {CONF_TAU:.2f}" if conf_thresh
            else f"budget=(lo {lo_frac:.4f}, hi {hi_frac:.4f})")
    log(f"  {tag}trainable {info['n_trainable_params']:,} params; "
        f"BN frozen {n_bn}; {rule}"
        f"{'' if use_cons else '; consistency term OFF'}")
    opt = torch.optim.Adam(trainable, lr=TTA_LR)
    n = len(buf)
    for ep in range(epochs):
        # rescore the pool with the current model
        model.eval()
        s = []
        with torch.no_grad():
            for i in range(0, n, BATCH):
                x = buf[i:i + BATCH].to(DEVICE, non_blocking=True).float()
                with P.amp_ctx(DEVICE):
                    s.append(torch.softmax(model(x)[0].float(), 1)[:, fake_col].cpu())
        s = torch.cat(s).numpy()
        pl = torch.full((n,), -1, dtype=torch.long)
        if conf_thresh:
            pl[torch.from_numpy(s <= 1.0 - CONF_TAU)] = 0
            pl[torch.from_numpy(s >= CONF_TAU)] = 1
        else:
            if lo_frac > 0:
                pl[torch.from_numpy(s <= np.quantile(s, lo_frac))] = 0
            if hi_frac > 0:
                pl[torch.from_numpy(s >= np.quantile(s, 1 - hi_frac))] = 1
        model.train()
        P.freeze_batchnorm(model)
        order = torch.randperm(n)
        for i in range(0, n, BATCH):
            sel = order[i:i + BATCH]
            x = buf[sel].to(DEVICE, non_blocking=True).float()
            bpl = pl[sel].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with P.amp_ctx(DEVICE):
                logits = model(x)[0]
                # fake_col may be 0; map to a canonical [real, fake] ordering so
                # cross-entropy against pseudo-labels means the same thing on
                # every checkpoint.
                lg = logits if fake_col == 1 else logits.flip(1)
                p = torch.softmax(lg, 1)
                loss = torch.zeros((), device=DEVICE)
                conf = bpl >= 0
                if conf.any():
                    loss = loss + F.cross_entropy(lg[conf], bpl[conf])
                if use_cons:
                    lga = model(P.augment(x))[0]
                    lga = lga if fake_col == 1 else lga.flip(1)
                    loss = loss + LAMBDA_CONS * F.mse_loss(
                        torch.softmax(lga, 1), p.detach())
            if loss.requires_grad:
                loss.backward()
                opt.step()
        log(f"  {tag}epoch {ep+1}/{epochs} done "
            f"(pseudo-labelled {int((pl >= 0).sum())}/{n})")
    return model


# ------------------------------------------------------- baseline TTA methods
# The point of running Tent / SHOT / ETA / SAR *here*, rather than only on the
# balanced leave-one-corpus-out grid, is that this is the pool that looks like
# deployment: 97.2% spoof. Every one of these methods was developed and
# evaluated on class-balanced benchmarks. If they degrade a state-of-the-art
# detector at realistic prevalence while a prior-corrected variant does not,
# then the balanced-pool assumption is a property of the *field's evaluation
# practice*, not a quirk of our own recipe.
def _canon_forward(model, buf, fake_col):
    """Callback for tta_baselines: pool indices -> logits in [real, fake] order."""
    def f(sel):
        x = buf[sel.cpu()].to(DEVICE, non_blocking=True).float()
        logits = model(x)[0]
        return logits if fake_col == 1 else logits.flip(1)
    return f


def run_baseline(arm, model, buf, crop, fake_col, epochs=TTA_EPOCHS, tag=""):
    params, info = P.set_tta_params(model)
    if arm == "shot":
        # SHOT keeps the source hypothesis and moves only the feature
        # extractor. Freezing the head keeps that contrast real rather than
        # turning SHOT into a re-parameterisation of ours.
        frozen = 0
        for mod_name, mod in model.named_modules():
            if mod_name.split(".")[-1] in ("out_layer", "classifier", "projector"):
                for q in mod.parameters():
                    q.requires_grad_(False)
                    frozen += 1
        params = [q for q in model.parameters() if q.requires_grad]
        log(f"  {tag}SHOT: head frozen ({frozen} tensors), {len(params)} left")
    n_bn = P.freeze_batchnorm(model)
    log(f"  {tag}{arm}: {sum(q.numel() for q in params):,} trainable params, "
        f"BN frozen {n_bn}")
    fwd = _canon_forward(model, buf, fake_col)
    n = len(buf)
    kw = dict(epochs=epochs, lr=TTA_LR, batch=BATCH, log=log)
    if arm == "tent":
        opt = torch.optim.Adam(params, lr=TTA_LR)
        for ep in range(epochs):
            model.train(); P.freeze_batchnorm(model)
            order = torch.randperm(n)
            for i in range(0, n, BATCH):
                opt.zero_grad(set_to_none=True)
                with P.amp_ctx(DEVICE):
                    pr = torch.softmax(fwd(order[i:i + BATCH]).float(), 1)
                    loss = -(pr * torch.log(pr + 1e-8)).sum(1).mean()
                loss.backward(); opt.step()
            log(f"  {tag}tent epoch {ep+1}/{epochs}")
        return model
    def mode_fn(train):
        model.train() if train else model.eval()
        P.freeze_batchnorm(model)      # never let BN statistics move

    mode_fn(True)
    if arm == "shot":
        return TB.shot(model, n, fwd, params, lambda: P.amp_ctx(DEVICE), DEVICE,
                       mode_fn=mode_fn, **kw)
    if arm == "eta":
        return TB.eata(model, n, fwd, params, lambda: P.amp_ctx(DEVICE), DEVICE, **kw)
    if arm == "sar":
        snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return TB.sar(model, n, fwd, params, lambda: P.amp_ctx(DEVICE), DEVICE,
                      snapshot=lambda: snap,
                      restore=lambda sn: model.load_state_dict(sn), **kw)
    raise ValueError(arm)


BASELINE_ARMS = ("tent", "shot", "eta", "sar")


# ------------------------------------------------------------------ reporting
def report(ckpt_name, method, setting, ev, scores, adapt_utts=None, extra=None):
    if setting == "disjoint_eval" and adapt_utts is not None:
        mask = ~ev.utt.isin(adapt_utts).values
        sub, sc = ev[mask], scores[mask]
    else:
        sub, sc = ev, scores
    y = sub.label.values.astype(int)
    # A slice can be single-class -- the disjoint half of a small pool at 97%
    # spoof easily contains no bona fide at all -- and then the ROC is
    # undefined and compute_eer raises "All-NaN slice". One degenerate slice
    # must cost that row, not the remaining hours of the stage.
    if len(np.unique(y)) < 2:
        log(f"  !! {ckpt_name} {method} [{setting}]: slice has one class only "
            f"({len(sub)} clips, all label {int(y[0])}) -- no EER/AUC, row skipped")
        return None
    eer, _ = compute_eer(y, sc)
    auc = float(roc_auc_score(y, sc))
    acc = float(accuracy_score(y, (sc >= 0.5).astype(int))) * 100
    # label-free median-threshold control: what a one-line rule would deliver
    acc_med = float(accuracy_score(y, (sc >= np.median(sc)).astype(int))) * 100
    row = dict(ckpt=ckpt_name, method=method, setting=setting,
               seed=SEED, eval_sub=EVAL_SUB or 0, q=Q, skew_target=SKEW or 0.0,
               eer=round(eer * 100, 3), auc=round(auc, 4), acc=round(acc, 3),
               acc_median_rule=round(acc_med, 3), n=len(sub),
               eer_operating_accuracy=round(100 - eer * 100, 3),
               eer_operating_gap=round((100 - eer * 100) - acc, 3), smoke=SMOKE)
    diagnostic = threshold_diagnostics(y, sc)
    row.update(oracle_accuracy=round(100 * diagnostic["oracle_accuracy"], 3),
               threshold_gap=round(100 * diagnostic["threshold_gap"], 3))
    if extra:
        row.update(extra)
    record(**row)
    log(f"  >>> {ckpt_name} {method} [{setting}] EER {row['eer']:.3f} "
        f"AUC {row['auc']:.4f} acc {row['acc']:.2f} "
        f"(median-rule {row['acc_median_rule']:.2f}, "
        f"oracle threshold gap {row['threshold_gap']:.2f})")
    return row


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    log(f"=== published checkpoints on {CORPUS} | smoke={SMOKE} | seed={SEED} "
        f"| eval_sub={EVAL_SUB} | ckpts={CKPTS} | arms={ARMS} ===")
    ev = build_eval()
    log(f"[{CORPUS}] eval clips on disk: {len(ev)}  "
        f"spoof {ev.label.mean()*100:.2f}%")
    if MAX_EVAL:
        ev = ev.sample(MAX_EVAL, random_state=0).reset_index(drop=True)
        log(f"SMOKE: subsampled to {len(ev)}")
    bbse_pools = {}
    for kind in set(BBSE_SOURCE.values()):
        pool_df = build_bbse_source(kind)
        if pool_df is not None:
            bbse_pools[kind] = pool_df
            log(f"BBSE source pool '{kind}': {len(pool_df)} clips "
                f"({pool_df.label.value_counts().to_dict()})")

    if SKEW:
        rng_s = np.random.RandomState(4242)
        f = ev.index[ev.label == 1].values
        r = ev.index[ev.label == 0].values
        n_real = int(round(len(f) * (1 - SKEW) / SKEW))
        if n_real > len(r):           # real-limited: thin the fake class instead
            n_real = len(r)
            f = rng_s.choice(f, int(round(n_real * SKEW / (1 - SKEW))), replace=False)
        keep = np.concatenate([f, rng_s.choice(r, n_real, replace=False)])
        ev = ev.loc[np.sort(keep)].reset_index(drop=True)
        log(f"PUBA_SKEW={SKEW}: resampled to {len(ev)} clips, "
            f"P(fake)={ev.label.mean():.4f} "
            f"({int(ev.label.sum())} fake / {int((1-ev.label).sum())} real); "
            f"min(pi,1-pi)={min(ev.label.mean(), 1-ev.label.mean()):.4f}, "
            f"so Eq.(3) needs q <= that")
    if EVAL_SUB:
        # fixed across seeds and arms, so every comparison stays paired
        ev = ev.sample(min(EVAL_SUB, len(ev)), random_state=12345)
        ev = ev.sort_values("utt").reset_index(drop=True)
        log(f"PUBA_EVAL_SUB: scoring a fixed {len(ev)}-trial subset "
            f"(spoof {ev.label.mean()*100:.2f}%) -- NOT the official pooled number")
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    adapt_idx = rng.choice(len(ev), min(ADAPT_N, len(ev)), replace=False)
    adapt_df = ev.iloc[np.sort(adapt_idx)].reset_index(drop=True)
    log(f"adapt pool (unlabeled) {len(adapt_df)} clips; "
        f"true spoof rate {adapt_df.label.mean()*100:.2f}% (NOT used)")

    already = done_rows()
    for name in CKPTS:
        cfg = P.CHECKPOINTS[name]
        crop, fake_col = cfg["crop"], cfg["fake_col"]
        log(f"--- {name} | {cfg['arch']} | crop {crop} | fake_col {fake_col} ---")
        base, nf, nb = P.build_model(name, DEVICE, log=log)
        log(f"  loaded {nf} front-end + {nb} back-end tensors")

        # source scores over the full official eval (cached to disk)
        stag = ("" if (SEED == 0 and not EVAL_SUB and abs(Q - 0.3) < 1e-9
                       and not SKEW)
                else f"__s{SEED}_e{EVAL_SUB or 0}_q{Q}_k{SKEW or 0}")
        spath = f"{SCORES_DIR}/{name}__source{stag}.npy"
        if os.path.exists(spath):
            s_src = np.load(spath)
            log(f"  reusing cached source scores {spath}")
        else:
            s_src = stream_score(base, ev.path.tolist(), crop, fake_col,
                                 tag=f"{name}/source ")
            np.save(spath, s_src)
        if _key(name, "source", "official_eval") not in already and "source" in ARMS:
            report(name, "source", "official_eval", ev, s_src)
            report(name, "source", "disjoint_eval", ev, s_src, set(adapt_df.utt))

        # decode the adapt pool once, reuse for every adaptation arm
        need_adapt = [a for a in ARMS if a != "source"
                      and _key(name, a, "official_eval") not in already]
        if not need_adapt:
            del base
            torch.cuda.empty_cache()
            continue
        log(f"  decoding adapt pool ({len(adapt_df)} clips) ...")
        abuf = decode_batch(adapt_df.path.tolist(), crop)
        s_adapt_src = s_src[adapt_df.index.values] if False else None

        for arm in need_adapt:
            model = copy.deepcopy(base)
            extra = {}
            if arm == "ours_fixed":
                lo, hi = Q, Q
            elif arm in ("ours_gmm", "ours_oracle"):
                # Same machinery, different source of the prior, so the cost of
                # the ESTIMATOR is isolated from the cost of the correction.
                # ours_oracle uses the true labels of the adapt pool and is
                # reported as an upper bound, never as a method.
                with torch.no_grad():
                    model.eval()
                    s_pool = []
                    for i in range(0, len(abuf), BATCH):
                        x = abuf[i:i + BATCH].to(DEVICE).float()
                        with P.amp_ctx(DEVICE):
                            s_pool.append(torch.softmax(
                                model(x)[0].float(), 1)[:, fake_col].cpu())
                    s_pool = torch.cat(s_pool).numpy()
                if arm == "ours_gmm":
                    pi_fake = gmm_pi_fake(s_pool)
                else:
                    pi_fake = float(adapt_df.label.mean())
                lo, hi = tail_budget(Q, 1.0 - pi_fake)
                extra = dict(pi_fake_hat=round(pi_fake, 4),
                             pi_fake_true=round(float(adapt_df.label.mean()), 4),
                             lo_frac=round(lo, 4), hi_frac=round(hi, 4))
                log(f"  [{arm}] pi_fake={pi_fake:.4f} "
                    f"(true {adapt_df.label.mean():.4f}) budget=({lo:.3f},{hi:.3f})")
            elif arm in ("ours_bbse", "ours_bbse_nocons"):
                src_kind = BBSE_SOURCE.get(name)
                if src_kind not in bbse_pools:
                    log(f"  [{arm}] no labelled training corpus on disk for "
                        f"{name} -- skipping (BBSE cannot be run honestly "
                        f"without the checkpoint's own source data)")
                    del model
                    torch.cuda.empty_cache()
                    continue
                # target prediction histogram from the FROZEN source model on
                # the adapt pool only (no labels, no eval-set leakage beyond the
                # unlabeled pool the method already sees).
                with torch.no_grad():
                    s_pool = []
                    model.eval()
                    for i in range(0, len(abuf), BATCH):
                        x = abuf[i:i + BATCH].to(DEVICE).float()
                        with P.amp_ctx(DEVICE):
                            s_pool.append(torch.softmax(
                                model(x)[0].float(), 1)[:, fake_col].cpu())
                    s_pool = torch.cat(s_pool).numpy()
                pi_fake, M, q = bbse_pi_fake(model, crop, fake_col,
                                             bbse_pools[src_kind], s_pool)
                lo, hi = tail_budget(Q, 1.0 - pi_fake)
                extra = dict(pi_fake_hat=round(pi_fake, 4),
                             pi_fake_true=round(float(adapt_df.label.mean()), 4),
                             lo_frac=round(lo, 4), hi_frac=round(hi, 4))
                log(f"  BBSE pi_fake_hat={pi_fake:.4f} (true "
                    f"{adapt_df.label.mean():.4f})  M={M.round(3).tolist()} "
                    f"q={q.round(3).tolist()}")
            elif arm == "ours_conf":
                # prior-free: no quantiles at all, a fixed confidence threshold
                lo = hi = None
                extra = dict(conf_tau=CONF_TAU)
            elif arm in BASELINE_ARMS:
                lo = hi = None
            else:
                raise ValueError(arm)
            log(f"  [{arm}] adapting ...")
            ta = time.time()
            if arm in BASELINE_ARMS:
                model = run_baseline(arm, model, abuf, crop, fake_col,
                                     tag=f"{arm} ")
            else:
                # ours_bbse_nocons isolates which half of the objective produces
                # the DF gain: same prior-corrected pseudo-label budget, no
                # channel-consistency term.
                model = adapt(model, abuf, crop, fake_col, lo, hi,
                              use_cons=(arm != "ours_bbse_nocons"),
                              conf_thresh=(arm == "ours_conf"),
                              tag=f"{arm} ")
            extra["adapt_min"] = round((time.time() - ta) / 60, 1)
            s = stream_score(model, ev.path.tolist(), crop, fake_col,
                             tag=f"{name}/{arm} ")
            np.save(f"{SCORES_DIR}/{name}__{arm}{stag}.npy", s)
            report(name, arm, "official_eval", ev, s, extra=extra)
            report(name, arm, "disjoint_eval", ev, s, set(adapt_df.utt), extra=extra)
            del model
            torch.cuda.empty_cache()
        del abuf, base
        torch.cuda.empty_cache()

    log(f"DONE in {(time.time()-t0)/60:.1f} min | peak GPU "
        f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
