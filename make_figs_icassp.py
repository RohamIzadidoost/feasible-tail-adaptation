"""Figures for main_icassp.tex, drawn only from committed result CSVs.

fig_mechanism.pdf (full width)
  (a) minority-tail pseudo-label purity under the symmetric budget and under FTA,
      with the counting bound      <- manuscript_audit/initial_tails.csv
  (b) accuracy after adaptation vs the worst-tail purity bound, every adapted run
      on DF and In-the-Wild (q sweep, prevalence intervention, prior-split arms)
                                     <- results_protocol_a_public{,_itw}.csv
  (c) In-the-Wild resampled to a target P(fake): accuracy per rule
                                     <- results_protocol_a_public_itw.csv
  (d) official DF, nine cells: EER vs accuracy for source, symmetric and FTA
                                     <- manuscript_audit/scores.csv (audited)

fig_breadth.pdf (one column)
  the two label-free guards on DF: rank agreement vs predicted-rate drift; the
  369-cell breadth numbers are printed to stdout for the text
                                     <- rank_monitor_df2021.csv

    python make_figs_icassp.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
})

C_SRC, C_SYM, C_OURS, C_ORA = "#555555", "#c0392b", "#1f5fa8", "#2e8b57"
PI_FAKE = {"DF": 0.9722, "ITW": 0.3718}


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontsize=9, pad=3)


def worst_bound(pi_fake, q_real, q_fake):
    """Worst-tail purity bound min over classes of min(1, pi_c / q_c)."""
    return min(1.0, (1 - pi_fake) / q_real, pi_fake / q_fake)


# ------------------------------------------------------------------ data
pur = pd.read_csv("pseudolabel_purity.csv")
df = pd.read_csv("results_protocol_a_public.csv")
df = df[df.setting == "official_eval"].copy()
itw = pd.read_csv("results_protocol_a_public_itw.csv")
itw = itw[itw.setting == "official_eval"].copy()

# every adapted run with a known budget -> (bound, accuracy)
law = []
for _, r in df.iterrows():
    q = 0.3 if np.isnan(r.q) else r.q
    if r.method == "ours_fixed":
        law.append(("DF", "symmetric", worst_bound(PI_FAKE["DF"], q, q), r.acc))
    elif r.method == "ours_bbse":
        pf = r.pi_fake_hat
        law.append(("DF", "prior-split", worst_bound(PI_FAKE["DF"], 2 * q * (1 - pf), 2 * q * pf), r.acc))
for _, r in itw.iterrows():
    q = 0.3 if np.isnan(r.q) else r.q
    pf_true = PI_FAKE["ITW"] if np.isnan(r.skew_target) else r.skew_target
    if r.method == "ours_fixed":
        law.append(("ITW", "symmetric", worst_bound(pf_true, q, q), r.acc))
    elif r.method in ("ours_bbse", "ours_gmm", "ours_oracle"):
        pf = r.pi_fake_hat
        law.append(("ITW", "prior-split", worst_bound(pf_true, 2 * q * (1 - pf), 2 * q * pf), r.acc))
law = pd.DataFrame(law, columns=["corpus", "rule", "bound", "acc"])

# ------------------------------------------------------------------ fig 1
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.36),
                         gridspec_kw=dict(width_ratios=[1.05, 1, 1, 1]))

# (a) realized minority-tail purity, symmetric vs FTA (audited replay)
ax = axes[0]
tails = pd.read_csv("manuscript_audit/initial_tails.csv")
tails["minority"] = np.where(tails.corpus == "df2021", 0, 1)
t = tails[tails.label == tails.minority]
xs = {"symmetric": 0, "bbse": 1}
for corpus, col, mk, lab in [("df2021", C_SYM, "o", "DF (97.2% spoof)"),
                             ("itw", C_OURS, "^", "In-the-Wild (37.2% fake)")]:
    g = t[t.corpus == corpus]
    for rule in ("symmetric", "bbse"):
        r = g[g.rule == rule]
        ax.plot(np.full(len(r), xs[rule]) + (0.04 if corpus == "itw" else -0.04),
                r.purity, mk, color=col, ms=3.6, mew=0.7, mfc="none", ls="none",
                label=lab if rule == "symmetric" else None)
    for _, row in g[g.rule == "symmetric"].iterrows():
        o = g[(g.rule == "bbse") & (g.ckpt == row.ckpt)]
        if len(o):
            ax.plot([xs["symmetric"] + (0.04 if corpus == "itw" else -0.04),
                     xs["bbse"] + (0.04 if corpus == "itw" else -0.04)],
                    [row.purity, o.purity.iloc[0]], color=col, lw=0.5, alpha=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["symmetric $q$", "FTA"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(0, 1.06)
ax.set_ylabel("minority-tail purity")
ax.text(0.5, 0.62, "In-the-Wild", ha="center", fontsize=9, color=C_OURS)
ax.text(0.5, 0.26, "DF", ha="center", fontsize=9, color=C_SYM)
ax.set_ylim(0, 1.12)
panel_label(ax, "(a) the budget fixes purity")

# (b) accuracy vs worst-tail bound, every adapted run
ax = axes[1]
for (corpus, rule), g in law.groupby(["corpus", "rule"]):
    col = C_SYM if rule == "symmetric" else C_OURS
    mk = "o" if corpus == "DF" else "^"
    ax.plot(g.bound, g.acc, mk, color=col, ms=3.4, mew=0.7,
            mfc=col if corpus == "DF" else "none", ls="none",
            label=f"{corpus}, {rule}")
ax.set_xlim(0, 1.06)
ax.set_ylim(40, 102)
ax.set_xlabel("worst-tail purity bound")
ax.set_ylabel("accuracy (%)", labelpad=1)
panel_label(ax, f"(b) {len(law)} adapted runs")

# (c) In-the-Wild prevalence intervention
ax = axes[2]
native = itw[itw.skew_target.isna() & (itw.ckpt == "ssl_aasist_wavefake")]
skew = itw[itw.skew_target.notna()]


def series(method, q=None):
    pts = []
    n = native[native.method == method]
    if q is None or q == 0.3:
        if len(n):
            pts.append((PI_FAKE["ITW"], n.acc.iloc[0]))
    s = skew[skew.method == method]
    if q is not None:
        s = s[np.isclose(s.q, q)]
    pts += list(zip(s.skew_target, s.acc))
    return np.array(sorted(pts))


for method, q, lab, col, ls, mk in [
        ("source", None, "source", C_SRC, "-", "s"),
        ("ours_fixed", 0.3, "sym. $q{=}0.3$", C_SYM, "-", "o"),
        ("ours_bbse", None, "FTA", C_OURS, "-", "^")]:
    p = series(method, q)
    ax.plot(p[:, 0], p[:, 1], ls=ls, marker=mk, color=col, ms=3,
            mfc="none" if ls != "-" else col, mew=0.7, label=lab)
ax.axvline(PI_FAKE["ITW"], color="#999999", lw=0.6, ls=":")
ax.text(0.388, 43, "native", fontsize=9, color="#777777", ha="left", va="bottom")
ax.set_xlim(0.33, 1.0)
ax.set_ylim(40, 102)
ax.set_xlabel("resampled $P(\\mathrm{fake})$")
ax.set_ylabel("accuracy (%)")
ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.02), handletextpad=0.3,
          borderpad=0.2, labelspacing=0.1, framealpha=0.0, handlelength=1.4)
panel_label(ax, "(c) In-the-Wild, prevalence set")

# (d) official DF, nine cells, audited metrics
ax = axes[3]
aud = pd.read_csv("manuscript_audit/scores.csv")
aud = aud[(aud.corpus == "df2021") & (aud.setting == "available_pool") & (aud.q == 0.3)]
piv = aud.pivot_table(index=["ckpt", "seed", "eval_sub"], columns="arm",
                      values=["eer", "accuracy"])
piv = piv.dropna(subset=[("eer", "ours_bbse"), ("eer", "ours_fixed"), ("eer", "source")])
for c in piv.index:
    ax.plot([piv.loc[c, ("eer", "ours_fixed")], piv.loc[c, ("eer", "ours_bbse")]],
            [piv.loc[c, ("accuracy", "ours_fixed")], piv.loc[c, ("accuracy", "ours_bbse")]],
            color="#bbbbbb", lw=0.5, zorder=1)
for m, col, mk in [("source", C_SRC, "s"), ("ours_fixed", C_SYM, "o"),
                   ("ours_bbse", C_OURS, "^")]:
    ax.plot(piv[("eer", m)], piv[("accuracy", m)], mk, color=col, ms=3.6, mew=0.7,
            zorder=2)
ax.text(5.1, 93, "source, FTA", fontsize=9, color="#333333", ha="center")
ax.text(5.5, 68, "symmetric", fontsize=9, color=C_SYM, ha="center")
ax.set_xlabel("EER (%)")
ax.set_ylabel("accuracy (%)")
ax.set_ylim(55, 103)
ax.set_xlim(3.0, 6.9)
panel_label(ax, f"(d) DF, {len(piv)} cells")

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout(pad=0.4, w_pad=0.6)
fig.savefig("fig_mechanism.pdf")
plt.close(fig)

# ------------------------------------------------------------------ fig 2
res = pd.read_csv("results_public_ckpt_multi.csv")
t = res[res.setting == "transductive"]
key = ["family", "target", "seed"]
s = t[t.method == "source"].groupby(key).mean(numeric_only=True)
o = t[t.method == "ours"].groupby(key).mean(numeric_only=True)
m = s.join(o, lsuffix="_s", rsuffix="_o", how="inner")
m["ds"] = 100 - m.eer_s - m.acc_s
m["do"] = 100 - m.eer_o - m.acc_o
print(f"third-party cells: {len(m)}, deficit {m.ds.mean():.2f} -> {m.do.mean():.2f}, "
      f"improved {(m.do < m.ds).sum()}")

fig, ax = plt.subplots(figsize=(3.3, 1.25))
rm = pd.read_csv("rank_monitor_df2021.csv")
rm = rm[rm.arm != "source"]
harm = (rm.d_acc < -1) | (rm.d_eer < -1) | (rm.d_auc < -0.01)
names = {"eta": "ETA", "tent": "Tent", "shot": "IM-PL", "ours_fixed": "sym.",
         "ours_bbse": "FTA", "sar": "SAR"}
ax.axvspan(0.75, 1.02, ymax=0.12 / 0.44, color=C_OURS, alpha=0.10, lw=0)
ax.text(0.885, 0.085, "accept", ha="center", fontsize=7, color=C_OURS)
for (_, r), h in zip(rm.iterrows(), harm):
    ax.plot(r.rho_src, r.prior_gap, "x" if h else "o", color=C_SYM if h else C_OURS,
            ms=3.6, mew=0.9, mfc="none")
    if h:
        dx, ha = ((4, "left") if r.arm in ("tent",) or r.rho_src > 0.8
                  else (-4, "right"))
        ax.annotate(names[r.arm], (r.rho_src, r.prior_gap), fontsize=7,
                    xytext=(dx, 3), textcoords="offset points", ha=ha,
                    color="#444444")
ax.set_xlim(0.25, 1.02)
ax.set_ylim(-0.02, 0.44)
ax.set_xlabel(r"rank agreement $\rho$ with the frozen source")
ax.set_ylabel("rate drift")
ax.grid(alpha=0.3)
fig.tight_layout(pad=0.25, w_pad=0.5)
fig.savefig("fig_breadth.pdf")
print("wrote fig_mechanism.pdf, fig_breadth.pdf;", len(law), "runs in panel (b)")
