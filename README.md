# Feasible-Tail Adaptation (FTA)

Code, cached scores and manuscript for **"Feasible-Tail Test-Time Adaptation for
Audio Deepfake Detection under Prevalence Shift"**.

Confident-tail test-time adaptation pseudo-labels the extreme quantiles of a
detector's scores on unlabelled target audio and self-trains on them. That rule
is infeasible wherever the deployment pool is class-skewed: a counting bound caps
pseudo-label purity at `min(1, pi_c / q_c)` for any detector, so a symmetric
budget `q` needs `q <= min(pi, 1-pi)`. FTA sizes each tail to the class it is
meant to contain, `q_c = 2 q pi_hat_c`, from a label-free prior estimate, capped
by a prevalence bound where no estimate can be trusted. At `pi_hat = 1/2` it is
identical to the symmetric rule, so it generalises the recipe it repairs.

## Reproducing the paper without a GPU

Everything in Tables 1 and 2 and in Figure 1 is recomputed from the cached score
arrays and the per-fold result files in this repository. No audio, no model
weights and no GPU are needed.

```bash
pip install -r requirements.txt
python make_tables_icassp.py     # every number in Tables 1 and 2, and Secs. 4.1 / 4.7
python make_figs_icassp.py       # regenerates fig_mechanism.pdf
python audit_icassp.py           # recomputes metrics from the raw .npy score arrays
python -m unittest test_metrics  # regression tests for the EER estimator
```

`make_tables_icassp.py` prints the table bodies directly; compare them against
`main_icassp.pdf`. `audit_icassp.py` additionally needs the official DF trial
metadata and the original manifests, and writes the files already provided under
`manuscript_audit/`.

## What is here

| Path | Contents |
|---|---|
| `main_icassp.tex`, `main_icassp.pdf` | manuscript (ICASSP template, `spconf.sty`) |
| `make_tables_icassp.py` | regenerates every reported number from the CSVs |
| `make_figs_icassp.py` | regenerates Figure 1 |
| `protocol_a_public.py` | Protocol A/C driver: FTA and the baselines on released checkpoints |
| `public_ckpt_tta.py` | adaptation loop, tail selection, scoring, checkpoint loading |
| `tta_baselines.py` | Tent, ETA/EATA, SAR and the IM-PL baseline |
| `metrics.py`, `test_metrics.py` | EER with interpolated ROC crossing, threshold diagnostics, tests |
| `audit_icassp.py` | recomputes all metrics from cached scores and records provenance |
| `scores_protocol_a_public*/` | per-arm score arrays (93 files) behind every DF and In-the-Wild number |
| `manuscript_audit/` | audited metrics, realised tail purities, threshold controls, provenance hashes |
| `results_*.csv` | per-fold and per-cell results for Protocol B and the third-party breadth arm |

## Re-running adaptation (needs a GPU and the corpora)

`protocol_a_public.py` adapts released checkpoints on unlabelled target audio.
It expects the ASVspoof2021-DF eval partition with the official CM keys, the
In-the-Wild benchmark, and the released checkpoints named in the paper. Corpora
and checkpoints are distributed by their own authors under their own terms and
are not redistributed here.

```bash
PUBA_CORPUS=df2021 python protocol_a_public.py      # official DF protocol
PUBA_CORPUS=itw    python protocol_a_public.py      # In-the-Wild
```

Useful environment variables: `PUBA_Q` (tail budget), `PUBA_ARMS`, `PUBA_CKPTS`,
`PUBA_SEED`, `PUBA_EVAL_SUB`, `PUBA_SKEW` (prevalence intervention).

## Citation

```bibtex
@inproceedings{izadidoost2027fta,
  title     = {Feasible-Tail Test-Time Adaptation for Audio Deepfake Detection
               under Prevalence Shift},
  author    = {Izadidoost, Roham and Sharma, Shweta and Srivastava, Sumit},
  booktitle = {Submitted to IEEE ICASSP},
  year      = {2027}
}
```

## Licence

Code and result files in this repository are released under the MIT licence (see
`LICENSE`). `spconf.sty` is the ICASSP paper-kit style file and is redistributed
unmodified under its own terms. Audio corpora and third-party model checkpoints
are **not** included and remain under the licences of their respective authors.
