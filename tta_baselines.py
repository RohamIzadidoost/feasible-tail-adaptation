"""Modern test-time-adaptation baselines: SHOT, ETA/EATA, SAR.

Why this exists (ICASSP_REVIEW_AND_PLAN.md, W3): the paper's only TTA
comparison point is Tent (ICLR 2021), whose instability on this data is one of
its headline observations. But Tent's instability is *known*, and the
literature's answer to it is precisely the family this paper's method belongs
to -- reliable-sample filtering (ETA/EATA), sharpness-aware entropy (SAR), and
information-maximisation with pseudo-labels (SHOT). Showing that Tent collapses
and then proposing a stabilised variant, without comparing against the
published stabilisations, is the single cheapest objection for a reviewer to
raise.

The three methods are implemented against a small callback protocol so the same
code runs against our own XLS-R detector (adaptive_pipeline.py) and against
third-party checkpoints (public_ckpt_tta.py), which have different module trees,
different score-column conventions and different batch loaders.

Protocol -- the caller supplies:
    n            : int, pool size
    forward(sel) : LongTensor of pool indices -> logits (B, 2) in CANONICAL
                   [real, fake] column order. The caller is responsible for
                   flipping a checkpoint whose native order is [spoof, bonafide].
    params       : list of tensors to optimise (already requires_grad_(True))
    amp          : context-manager factory for autocast
    device       : torch device string

Everything below is unlabeled. Labels never enter.

References
    SHOT  Liang, Hu & Feng, "Do we really need to access the source data?
          Source hypothesis transfer for unsupervised domain adaptation",
          ICML 2020.
    EATA  Niu et al., "Efficient test-time model adaptation without forgetting",
          ICML 2022.
    SAR   Niu et al., "Towards stable test-time adaptation in dynamic wild
          world", ICLR 2023.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

# Entropy threshold from EATA/SAR: E_0 = 0.4 * ln(C). C = 2 here, so 0.277.
# (Both papers use 0.4*ln(1000) for ImageNet; the 0.4 coefficient is the
# transferable part, ln C is the maximum entropy of the label space.)
N_CLASSES = 2
E0 = 0.4 * math.log(N_CLASSES)
# EATA's redundancy threshold on cosine similarity. The published value (0.05)
# is calibrated for C=1000, where two confident probability vectors for different
# classes are near-orthogonal and a similarity below 0.05 is common. At C=2 the
# rule is degenerate in either of two ways, measured over the binary simplex:
#   * if the running average is not itself near a vertex, NOTHING passes -- the
#     minimum attainable cosine is 0.71 at an EMA of (.5,.5), 0.39 at (.7,.3) and
#     0.11 at (.9,.1), all above 0.05 -- so ETA silently becomes a no-op after the
#     first batch and reads as "the baseline didn't help";
#   * once the EMA does reach a vertex (past ~(.97,.03), which a 97%-spoof pool
#     drives it to), the only samples that pass are those of the MINORITY
#     predicted class, so the filter stops meaning "non-redundant" and starts
#     meaning "minority".
# Neither is the mechanism the paper describes, so we run the reliability filter
# alone -- which is EATA's own published ablation (ETA without redundancy) --
# rather than invent a new threshold. Set use_redundancy=True to restore the
# literal published rule.
D_MARGIN = 0.05
SAM_RHO = 0.05           # SAR / SAM neighbourhood radius
# SAR's model-recovery trigger on the moving-average entropy loss. Published
# value 0.2 -- for ImageNet, where E0 = 0.4*ln(1000) = 2.76, so the reset
# threshold sits at 0.029*ln(C), far BELOW the reliability band and fires only
# under genuine collapse. Transplanted literally into C=2, where E0 = 0.277, the
# constant 0.2 lands INSIDE the reliable band: every reliable sample already has
# H < 0.277, so the moving average is routinely under 0.2 and the guard reverts
# the model on essentially every batch, freezing SAR at its initial weights
# (verified: 32 resets in 32 steps, zero net parameter movement). We keep SAR's
# relative position instead of its absolute constant: e0 = (0.2/ln 1000)*ln C.
SAR_RESET_E = (0.2 / math.log(1000)) * math.log(N_CLASSES)
FISHER_ALPHA = 2000.0    # EATA anti-forgetting weight (paper's default)


def _entropy(logits):
    p = torch.softmax(logits, 1)
    return -(p * torch.log(p + 1e-8)).sum(1)


# --------------------------------------------------------------------- SHOT
def shot(model, n, forward, params, amp, device, epochs=4, lr=1e-4, batch=32,
         beta=0.3, mode_fn=None, log=print):
    """Information maximisation + centroid pseudo-labels (SHOT).

    L = H(p) - H(E[p]) + beta * CE(p, centroid pseudo-labels)

    The diversity term -H(E[p]) is exactly what Tent lacks and is SHOT's stated
    defence against the collapse-to-one-class failure. Faithful to the original
    in freezing the classifier head: SHOT's premise is that the source
    hypothesis is kept and only the feature extractor moves. Our own method does
    the opposite (it adapts the head), so this is a genuine methodological
    contrast rather than a re-parameterisation of ours.
    """
    opt = torch.optim.Adam(params, lr=lr)
    # The per-epoch centroid pass must run with dropout OFF, or the pseudo-labels
    # are drawn from a stochastic model -- our own adapt() scores in eval() for
    # exactly this reason. `mode_fn(train)` lets the caller switch modes in a way
    # that preserves any BatchNorm freezing it has applied (a bare model.train()
    # would silently re-enable BN statistics updates, which is AdaBN, a
    # different adaptation mechanism).
    if mode_fn is None:
        def mode_fn(train):
            model.train() if train else model.eval()
    for ep in range(epochs):
        # centroid pseudo-labels over the whole pool, recomputed each epoch
        mode_fn(False)
        with torch.no_grad():
            probs = []
            for i in range(0, n, batch):
                sel = torch.arange(i, min(i + batch, n), device=device)
                with amp():
                    probs.append(torch.softmax(forward(sel).float(), 1).cpu())
            probs = torch.cat(probs)
        # 2-class self-labelling: centroid = probability-weighted mean of the
        # soft assignments; the pseudo-label is the nearer centroid. With C=2
        # this reduces to a data-driven threshold rather than the fixed 0.5.
        w = probs / (probs.sum(0, keepdim=True) + 1e-8)
        cent = (w * probs).sum(0)                       # (2,)
        pl = (probs - cent.unsqueeze(0)).abs().argmin(1).to(device)
        mode_fn(True)
        order = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            sel = order[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with amp():
                logits = forward(sel)
                p = torch.softmax(logits.float(), 1)
                ent = -(p * torch.log(p + 1e-8)).sum(1).mean()
                pm = p.mean(0)
                div = -(pm * torch.log(pm + 1e-8)).sum()
                loss = ent - div + beta * F.cross_entropy(logits.float(), pl[sel])
            loss.backward()
            opt.step()
        log(f"    shot epoch {ep+1}/{epochs}")
    return model


# ---------------------------------------------------------------- ETA / EATA
def compute_fisher(model, params, n_src, forward_src, amp, device, batch=32,
                   log=print):
    """Diagonal Fisher on labelled source data, for EATA's anti-forgetting term.

    EATA estimates it from source samples with the model's own predictions as
    targets (no labels needed), which is what we do here.
    """
    fisher = [torch.zeros_like(p) for p in params]
    cnt = 0
    for i in range(0, n_src, batch):
        sel = torch.arange(i, min(i + batch, n_src), device=device)
        with amp():
            logits = forward_src(sel)
            loss = F.cross_entropy(logits.float(), logits.float().argmax(1))
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for f, g in zip(fisher, grads):
            if g is not None:
                f += g.detach() ** 2 * len(sel)
        cnt += len(sel)
    fisher = [f / max(cnt, 1) for f in fisher]
    log(f"    fisher over {cnt} source clips")
    return fisher


def eata(model, n, forward, params, amp, device, epochs=4, lr=1e-4, batch=32,
         fisher=None, anchor=None, use_redundancy=False, log=print):
    """Entropy minimisation on reliable, non-redundant samples (ETA/EATA).

    Two filters, both from the paper:
      * reliability -- keep only samples with H(p) < E0; high-entropy samples
        are the ones whose gradients are noisy and drive collapse.
      * redundancy  -- keep only samples whose probability vector is not
        cosine-similar to a moving average of the ones already used.
    Kept samples are weighted by exp(E0 - H(x)), so confident samples count more.

    With `fisher` and `anchor` supplied this is full EATA (the anti-forgetting
    regulariser pulls the adapted weights back toward the source weights,
    weighted by the diagonal Fisher); without them it is ETA, the paper's own
    ablation.
    """
    opt = torch.optim.Adam(params, lr=lr)
    ema = None                      # moving average of used probability vectors
    n_used = n_seen = 0
    for ep in range(epochs):
        order = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            sel = order[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with amp():
                logits = forward(sel)
            lf = logits.float()
            ent = _entropy(lf)
            keep = ent < E0
            if use_redundancy and keep.any() and ema is not None:
                p = torch.softmax(lf[keep], 1)
                cos = F.cosine_similarity(p, ema.unsqueeze(0).expand_as(p), dim=1)
                sub = cos.abs() < D_MARGIN
                idxk = keep.nonzero(as_tuple=True)[0][sub]
                keep = torch.zeros_like(keep)
                keep[idxk] = True
            if not keep.any():
                continue
            w = torch.exp(E0 - ent[keep].detach()).clamp(max=10.0)
            loss = (w * ent[keep]).sum() / w.sum()
            if fisher is not None and anchor is not None:
                reg = sum((f * (p_ - a) ** 2).sum()
                          for f, p_, a in zip(fisher, params, anchor))
                loss = loss + FISHER_ALPHA * reg
            loss.backward()
            opt.step()
            with torch.no_grad():
                pk = torch.softmax(lf[keep], 1).mean(0)
                ema = pk if ema is None else 0.9 * ema + 0.1 * pk
            n_used += int(keep.sum())
        # A filter that admits almost nothing turns the baseline into a no-op
        # that looks like a negative result. Surface it rather than let it pass.
        rate = n_used / max(n * (ep + 1), 1)
        flag = "  !! reliability filter admits <1% -- baseline is near-inert" if rate < 0.01 else ""
        log(f"    eata epoch {ep+1}/{epochs}  reliable-sample rate {rate:.3f}{flag}")
    return model


# ------------------------------------------------------------------------ SAR
class _SAM:
    """Sharpness-aware minimisation wrapper (two-step), as used by SAR."""

    def __init__(self, params, base_opt, rho=SAM_RHO):
        self.params = list(params)
        self.opt = base_opt
        self.rho = rho
        self.e_w = None

    @torch.no_grad()
    def first_step(self):
        grads = [p.grad for p in self.params]
        norm = torch.norm(torch.stack([g.norm(p=2) for g in grads if g is not None]), p=2)
        scale = self.rho / (norm + 1e-12)
        self.e_w = []
        for p, g in zip(self.params, grads):
            if g is None:
                self.e_w.append(None)
                continue
            e = g * scale
            p.add_(e)
            self.e_w.append(e)

    @torch.no_grad()
    def second_step(self):
        for p, e in zip(self.params, self.e_w):
            if e is not None:
                p.sub_(e)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)


def sar(model, n, forward, params, amp, device, epochs=4, lr=1e-4, batch=32,
        snapshot=None, restore=None, log=print):
    """Sharpness-aware reliable entropy minimisation with model recovery (SAR).

    Three ingredients, all from the paper:
      * the same reliability filter as ETA (H < E0),
      * a SAM step, so the update lands in a flat region rather than a sharp one
        -- the paper's diagnosis of why plain entropy minimisation collapses,
      * a recovery scheme: if the moving average of the post-SAM entropy falls
        below `SAR_RESET_E` the model is reset to its pre-adaptation weights,
        an explicit collapse guard.

    `snapshot`/`restore` implement that reset; if they are not supplied the
    recovery scheme is disabled and that is logged, because a SAR without it is
    a materially weaker baseline and the difference must not be silent.
    """
    base = torch.optim.SGD(params, lr=lr, momentum=0.9)
    sam = _SAM(params, base)
    ema = None
    snap = snapshot() if snapshot is not None else None
    if snap is None:
        log("    sar: NO recovery snapshot supplied -- reset scheme disabled")
    n_reset = n_used = 0
    for ep in range(epochs):
        order = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            sel = order[i:i + batch]
            base.zero_grad(set_to_none=True)
            with amp():
                logits = forward(sel)
            ent = _entropy(logits.float())
            keep = ent < E0
            if not keep.any():
                continue
            n_used += int(keep.sum())
            ent[keep].mean().backward()
            sam.first_step()
            base.zero_grad(set_to_none=True)
            with amp():
                logits2 = forward(sel)
            ent2 = _entropy(logits2.float())
            keep2 = ent2 < E0
            if not keep2.any():
                # undo the ascent step and skip
                with torch.no_grad():
                    for p, e in zip(sam.params, sam.e_w):
                        if e is not None:
                            p.sub_(e)
                continue
            loss2 = ent2[keep2].mean()
            loss2.backward()
            sam.second_step()
            l = float(loss2.detach())
            ema = l if ema is None else 0.9 * ema + 0.1 * l
            if snap is not None and ema is not None and ema < SAR_RESET_E:
                restore(snap)
                ema = None
                n_reset += 1
        rate = n_used / max(n * (ep + 1), 1)
        flag = "  !! reliability filter admits <1% -- baseline is near-inert" if rate < 0.01 else ""
        log(f"    sar epoch {ep+1}/{epochs}  resets {n_reset}  "
            f"reliable-sample rate {rate:.3f}{flag}")
    return model
