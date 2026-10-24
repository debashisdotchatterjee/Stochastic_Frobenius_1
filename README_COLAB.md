# Stochastic Frobenius Research: Google Colab Replication Package

## Deliverables

- **`Stochastic_Frobenius_Colab.ipynb`**: fully self-contained notebook. It embeds the entire source code; no Drive mount, GitHub repository, or uploaded helper file is required.
- **`stochastic_frobenius_colab.py`**: the complete standalone research script (can also run locally).
- **`Stochastic_Frobenius_Full_Run_Outputs.zip`**: a full study generated with master random seed **20261023** (13 professional figures, in PNG/PDF/SVG, and 19 tables in CSV/LaTeX).
- **`main.tex`, `ref.bib`, `main.pdf`**: previous mathematical working manuscript and bibliographic source, included for convenience.

## Google Colab: what to do

1. Open Google Colab: <https://colab.research.google.com/>.
2. Choose **File → Upload notebook** and upload **`Stochastic_Frobenius_Colab.ipynb`**.
3. Select **Runtime → Run all**. It uses the default CPU runtime; GPU acceleration is not required.
4. The notebook writes and executes the source itself, prints validations and tables, previews important figures, and downloads **`stochastic_frobenius_outputs.zip`** automatically.
5. Return the **newly generated ZIP** for manuscript drafting, statistical interpretation, and peer-review-style checking.

Run-time inputs in the notebook:

- `RUN_PROFILE = "full"` for journal-style simulations (the default).
- `RUN_PROFILE = "quick"` for a short correctness check.
- `RANDOM_SEED = 20261023` for the exact illustrated seeded simulation. Change this for independent replication.

The ZIP includes `run_manifest.json` with software versions, seed, parameters, and checksums.

## The mathematics being checked

For fixed packet sizes `a=(5,6,7,8)` and iid selection probabilities `p=(.4,.3,.2,.1)`:

\[
S_k=X_1+\cdots+X_k,\qquad
u_n(p)=\Pr\{\exists k\ge0:S_k=n\},\qquad
F_\eta=\max\{n\ge1:u_n(p)<\eta\}.
\]

The exact renewal recurrence is `u[0]=1` and

\[
u_n=\sum_{j:a_j\le n}p_j u_{n-a_j},\quad n\ge1.
\]

The script independently verifies `Apéry=[0,6,7,8,14]`, `classical F=9`, `mu=6`, `renewal limit=1/6`, the minorization `delta=83/6250`, `q=2793/3125`, and all five thresholds below:

| eta | certified F_eta |
|:---:|---:|
| 0.10 | 9 |
| 0.12 | 15 |
| 0.14 | 21 |
| 0.15 | 27 |
| 0.16 | 33 |

A **global exact rational tail certificate** is used; merely scanning the first `N` integers is not sufficient to establish `F_eta`.

## Empirical study and interpretation

The software distinguishes four materially different kinds of quantities:

1. **Exact mathematical quantities:** Fraction-arithmetic recurrence, independent multinomial-composition enumeration, Markov transition-powers, and rigorous tail bound.
2. **Finite Monte Carlo approximations:** simulate iid paths, compare their Bernoulli hitting indicators with exact `u_n`, and compute Wilson binomial confidence intervals.
3. **Sampling-based estimation:** generate multinomial packet-frequency datasets; study delta-method variances, RMSE, coverage of **asymptotic** confidence intervals, and plug-in integer threshold recovery.
4. **Rigorous selective finite-sample intervals:** the Hoeffding + coupling theorem, sharpened with the exact lower matrix transition bound. These guarantee `P(report interval AND miss true F) <= beta`, **not** conditional coverage given reporting. The constructed intervals may be highly conservative. A bootstrap panel is explicitly marked as approximate, not coverage-certified.

**Crucial limitation:** The reported nonasymptotic confidence intervals can remain very wide even for large training datasets. This is a meaningful negative computational finding about the conservative certificate, not a software failure. It highlights the need for sharper uncertainty propagation as a potential theoretical contribution.

## Files in the output archive

- `figures/Fig01`—`Fig13`: journal-style figures, each as PNG (320 dpi), vector PDF, and vector SVG.
- `tables/`: 19 tables, each in CSV and a LaTeX `booktabs`-compatible table.
- `verification/`: exact identities and explicit proof-certificate parameters.
- `source/`: exact execution script and manuscript sources (where present).
- `README.txt`: interpretation safeguards and a description of the experimental design.
- `run_manifest.json`: software and seed reproducibility information with SHA256 hashes.

## Scientific scope

These are synthetic experiments under the stated model. 
