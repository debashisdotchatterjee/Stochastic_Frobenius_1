Statistical Inference for Stochastic Frobenius Frontiers in Four-Generator Renewal Processes
Reproducible research code, mathematical verification, simulation studies, and a survey-data-calibrated application
Author: Dr. Debashis Chatterjee  
Affiliation: Department of Statistics, Visva-Bharati University, Santiniketan, West Bengal, India  
Correspondence: debashis.chatterjee@visva-bharati.ac.in
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)
Overview
The classical Frobenius problem asks which nonnegative integers are representable by a fixed set of positive integer generators. Representability does not imply reliable attainment when generator sizes are sampled sequentially and randomly. This project investigates stochastic Frobenius frontiers: the last integer target whose exact-hitting probability falls below a prespecified reliability requirement.
The associated research develops exact finite certificates, sensitivity bounds, statistical procedures for an integer-valued frontier, and critical-boundary asymptotics. The computational work separates (i) exact mathematical verification, (ii) Monte Carlo assessment of inferential procedures, and (iii) an application calibrated to 2024 US Census American Community Survey (ACS) household microdata.
> **Manuscript status:** Research code and numerical outputs are available here. The paper should be cited as a manuscript unless and until a journal publication, preprint DOI, or other permanent bibliographic record is supplied. Repository availability is not a claim of peer-reviewed acceptance.
Mathematical framework
Let the four generators satisfy
$$
2\le a_1<a_2<a_3<a_4,\qquad \gcd(a_1,a_2,a_3,a_4)=1,
$$
with positive sampling probabilities $p_j$ summing to one. For independent increments $X_k\in{a_1,a_2,a_3,a_4}$, let
$$
S_0=0,\quad S_k=\sum_{i=1}^kX_i,\quad
u_n(\boldsymbol p)=\Pr_{\boldsymbol p}(S_k=n\text{ for some }k\ge0).
$$
The renewal recurrence is
$$
u_0=1,\qquad
u_n=\sum_{j=1}^4p_j,u_{n-a_j},\quad n\ge1,
$$
where $u_m=0$ for $m<0$. Writing $\mu=\sum_{j=1}^4p_ja_j$, the stochastic Frobenius frontier is
$$
F_\eta(\boldsymbol a,\boldsymbol p)
=\max{n\ge1:u_n(\boldsymbol p)<\eta},
\qquad 0<\eta<1/\mu.
$$
The model concerns independent sampling with replacement and unlimited availability of the four generator sizes. It does not automatically describe finite inventories, dependent arrivals, or observed recruitment sequences.
Main computational findings
Exact, certified four-generator example
For $\boldsymbol a=(5,6,7,8)$ and $\boldsymbol p=(0.4,0.3,0.2,0.1)$, the classical Frobenius number is 9. Exact rational calculations and a finite tail certificate give:
Reliability threshold $\eta$	Certified frontier $F_\eta$
0.10	9
0.12	15
0.14	21
0.15	27
0.16	33
The analysis uses renewal recurrences, numerical-semigroup calculations, Markov-state checks, exact arithmetic, Monte Carlo simulation, and inferential diagnostics. A finite scan by itself is not used as a proof that no larger unreliable target exists.
Critical-boundary behaviour
At $\eta=83/625$, the same model satisfies $u_{21}=\eta$, with population frontier $F_\eta=15$. The plug-in frontier has a non-Gaussian two-point limiting law on ${15,21}$, although smooth fixed-target renewal probabilities admit Gaussian asymptotics. Boundary experiments and bootstrap diagnostics are discussed in the research; the repository root does not currently provide a separate boundary-experiment program, so those results should not be assumed reproducible from the baseline script alone.
ACS household-recruitment illustration
The empirical study uses 2024 ACS one-year PUMS housing records from California (CA), Texas (TX), and New York (NY). It estimates survey-weighted probabilities of household sizes ${5,6,7,8}$ and uses them to calibrate a hypothetical independent, whole-household recruitment process.
The prespecified smooth reliability summary is
$$
R_s=\tfrac14\sum_{n\in{15,21,27,33}}u_n(\boldsymbol p_s).
$$
State	Estimated $R_s$	SDR standard error	$F_{0.10}$	$F_{0.12}$
California	0.2110	0.0021	19	19
Texas	0.2317	0.0024	19	24
New York	0.2185	0.0034	19	19
The primary design-based, two-degree-of-freedom omnibus Wald comparison of $R_s$ gives $W=36.976$ (asymptotic $p=9.35\times10^{-9}$); these numerical results are conditional on the data-processing conventions recorded in the archived analysis.
Interpretation: These are population-composition-calibrated model calculations, not observed frequencies from actual household recruitment campaigns. The Wald approximation requires appropriate design-based asymptotic and replicate-covariance assumptions. Replicate-weight frontier distributions are descriptive and are not automatically confidence intervals for a discontinuous frontier.
Repository files
File	Purpose
`Stochastic_Frobenius_Colab.ipynb`	Google Colab notebook for the core verification and simulation study.
`stochastic_frobenius_colab.py`	Standalone Python implementation of the core study.
`stochastic_frobenius_outputs.zip`	Archived core-run outputs: 13 figures (PNG/PDF/SVG), 19 tables, verification files and run manifest.
`stochastic_frobenius_ACS_realdata_v1.zip`	Archived ACS analysis: seven figures, 12 CSV tables, source script, diagnostics, source checksums and run manifest.
`README_COLAB.md`	Detailed instructions for the core simulation notebook.
`LICENSE`	MIT software license.
Note: The ACS analysis source is packaged inside `stochastic_frobenius_ACS_realdata_v1.zip`, under `diagnostics/append_acs_realdata_analysis.py`; it is not currently a separate top-level repository file. The archived outputs should not be confused with the underlying raw ACS microdata.
Reproducing the computations
Option A — Google Colab (recommended)
Open the core notebook in Google Colab and choose Runtime → Run all. The default full profile runs the exact checks and Monte Carlo study, creates figures and tables, and exports `stochastic_frobenius_outputs.zip`. No GPU is required.
Set `RUN_PROFILE = 'quick'` inside the source for a shorter smoke test, or `RUN_PROFILE = 'full'` for the complete experiment. The core study uses master seed `20261023` by default.
Option B — Run the core script locally
Use Python 3.10+ with NumPy, SciPy, pandas, and Matplotlib installed:
```bash
python -m pip install numpy scipy pandas matplotlib
python stochastic_frobenius_colab.py --profile quick --no-download
python stochastic_frobenius_colab.py --profile full --seed 20261023 --no-download
```
The code includes exact arithmetic and assertions, writes a run manifest, and preserves the random seed and generated numerical files.
Option C — Reproduce the ACS analysis
Download and extract `stochastic_frobenius_ACS_realdata_v1.zip`. Its `diagnostics/` directory includes the executed ACS Python source, variable/method notes, and metadata. Run the source from a writable directory, with internet access and the additional `requests` dependency:
```bash
python -m pip install numpy scipy pandas matplotlib requests
python stochastic_frobenius_ACS_realdata_v1/diagnostics/append_acs_realdata_analysis.py
```
The ACS script downloads official 2024 one-year household PUMS ZIP files for CA, TX, and NY from the US Census Bureau, reads `NP`, `WGTP`, and `WGTP1`–`WGTP80`, and applies the successive-difference replication variance factor $4/80$. It writes a new ACS ZIP with figures, tables, gradients, frontier certificates, and input-data SHA-256 checksums. Execution requires access to the Census download server; these raw input files are not committed to this repository.
Reproducibility qualification: The archived ACS run excludes three size-eligible records because its preprocessing requires nonnegative values for every replicate weight. Census replicate weights may be signed; therefore, this is a documented analysis restriction, not a general survey-data validity rule. A signed-replicate-weight sensitivity rerun should be performed before treating the current estimates as definitive.
Statistical and computational safeguards
Exact recurrence calculations and independently checked finite stopping certificates are separated from Monte Carlo approximations.
Sampling uncertainty for smooth $u_n$ and $R_s$ is distinguished from nonregular inference for integer-valued $F_\eta$.
The original multinomial confidence results do not transfer automatically to ACS complex-survey sampling; the real-data analysis uses replicate-weight calculations under design-based assumptions.
Pointwise uncertainty bands are not simultaneous confidence bands; nominal bootstrap/replicate distributions are not finite-sample certificates.
Each archived output includes run metadata; the ACS archive also reports input ZIP checksums and state-specific processing diagnostics.
Data provenance
American Community Survey, 2024 one-year Public Use Microdata Sample (PUMS), housing-unit files, US Census Bureau. Data and variable definitions: ACS PUMS. The underlying survey data are third-party public data; the repository's MIT license applies to the author's software and does not relicense Census data.
Citation
Until a formal paper DOI or permanent preprint identifier is available, cite the project as:
> Chatterjee, D. (2026). *Statistical Inference for Stochastic Frobenius Frontiers in Four-Generator Renewal Processes*. Research manuscript and reproducibility repository. <https://github.com/debashisdotchatterjee/Stochastic_Frobenius_1>.
A provisional BibTeX entry is:
```bibtex
@misc{chatterjee2026stochasticfrobenius,
  author       = {Chatterjee, Debashis},
  title        = {Statistical Inference for Stochastic Frobenius Frontiers
                  in Four-Generator Renewal Processes},
  year         = {2026},
  howpublished = {Research manuscript and reproducibility repository},
  url          = {https://github.com/debashisdotchatterjee/Stochastic_Frobenius_1},
  note         = {Accessed: insert access date}
}
```
Software license and research integrity
Source code is distributed under the MIT License, copyright © 2026 Dr. Debashis Chatterjee. Please retain the license notice in copies or adaptations of the software.
AI-use disclosure: OpenAI's ChatGPT assisted with grammar and syntax, LaTeX preparation, and Python coding and debugging. The author is responsible for the mathematical statements, statistical assumptions, data processing, citations, and scientific conclusions.
Correspondence: debashis.chatterjee@visva-bharati.ac.in
