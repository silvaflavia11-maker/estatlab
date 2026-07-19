---
title: 'EstatLab: an educational statistical analysis application in Brazilian Portuguese with automatic didactic interpretation'
tags:
  - Python
  - statistics
  - statistics education
  - statistical process control
  - measurement systems analysis
  - design of experiments
  - Brazilian Portuguese
authors:
  - name: Flávia Silva
    orcid: 0009-0009-8732-6903
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 8 July 2026
bibliography: paper.bib
---

# Summary

EstatLab is an open-source desktop application for statistical analysis with a
graphical interface entirely in Brazilian Portuguese, designed for teaching
and academic use. Its distinguishing feature is the automatic *didactic
interpretation* of every analysis: alongside the numerical output, the
application states the null and alternative hypotheses in plain language,
reports the test statistic and p-value, spells out the decision at the chosen
significance level ("since p = 0.003 < α = 0.05, H₀ is rejected: there is
evidence that..."), and warns about violated or unverified assumptions,
suggesting the appropriate diagnostic (e.g., normality tests for small
samples).

The application covers the span of an undergraduate-to-graduate applied
statistics curriculum, including domains rarely found together in open-source
educational tools: descriptive statistics and estimation, hypothesis tests,
ANOVA with multiple comparisons, regression (linear, logistic, Poisson,
ordinal, nominal, nonlinear, orthogonal and partial least squares),
nonparametric methods, contingency tables, probability distributions, power
and sample size, statistical process control charts, process capability,
measurement systems analysis (Gage R&R), design of experiments (factorial,
Plackett–Burman, response surface and Taguchi), time series, multivariate
methods, reliability/survival analysis, resampling, and equivalence testing.

All numerical procedures are delegated to established, peer-reviewed
scientific libraries — SciPy [@virtanen2020], statsmodels [@seabold2010],
NumPy [@harris2020], pandas [@mckinney2010], scikit-learn [@pedregosa2011]
and lifelines [@davidsonpilon2019] — with Matplotlib for graphics
[@hunter2007]. EstatLab adds the worksheet interface, the pedagogical layer
and the orchestration, and ships an automated validation suite of 147 tests
that check its outputs against reference implementations, independently
recomputed formulas and classical published values.

# Statement of need

Statistics education in Portuguese-speaking countries faces a tooling gap.
Commercial packages such as Minitab [@minitab] dominate quality-engineering
curricula but are costly for students and institutions, and their outputs
presuppose statistical maturity that introductory students are still
building. General-purpose open alternatives such as JASP [@jasp] and jamovi
[@jamovi] offer excellent interfaces, but their coverage of industrial
statistics (control charts, process capability, measurement systems analysis,
design of experiments) is limited, and their interfaces and outputs are not
localized for didactic use in Portuguese. Programming environments such as R
[@rcoreteam] and Python provide complete coverage but impose a programming
barrier that shifts course time from statistical reasoning to syntax.

EstatLab addresses this gap with three design commitments. First,
*interpretation as a first-class output*: every result is accompanied by an
explicit hypothesis statement, decision rule and contextual conclusion in
Portuguese, following recommendations that introductory instruction emphasize
statistical thinking over computation [@gaise2016]. Second, *breadth aligned
with quality-engineering teaching*: the tool covers the industrial statistics
sequence (SPC, capability, MSA, DOE) that open educational tools typically
omit, making it usable across basic statistics courses and Six Sigma-style
training alike. Third, *auditability*: because computations are delegated to
open, peer-reviewed libraries and covered by a public validation suite,
instructors and researchers can verify exactly what each procedure does —
in contrast with closed-source alternatives.

The intended audience comprises instructors and students of statistics and
quality engineering in Portuguese-speaking institutions, as well as
researchers who need documented, reproducible analyses with outputs in
Portuguese for reports and theses. EstatLab has been used as the analysis
tool in ongoing applied research, with results reported in manuscripts
currently under preparation, and is being validated in internal teaching use
by an academic team.

# Functionality and validation

EstatLab is built in Python with a Qt (PySide6) interface. Data live in a
spreadsheet-like worksheet with Excel interoperability (clipboard, CSV,
XLSX); analyses accumulate in a session panel exportable to HTML and PDF;
graphs update automatically when data change and export to common raster and
vector formats. Projects (worksheets plus session) are saved in a single
portable file. The numerical engine is a set of pure functions separated
from the interface, which makes the 147-test validation suite fast and
CI-friendly; tests compare results with direct library calls, hand-computed
formulas and classical published values (e.g., control-chart constants,
tolerance-interval factors, exact Poisson confidence limits).

# Acknowledgements

The author thanks the academic validation team for testing and feedback.
EstatLab was developed with the assistance of an AI pair programmer under
the author's direction; all statistical computations are delegated to the
peer-reviewed libraries cited above, and correctness is enforced by the
public automated validation suite.

# References
