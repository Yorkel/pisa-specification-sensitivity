# pisa-specification-sensitivity

[![tests](https://github.com/Yorkel/pisa-specification-sensitivity/actions/workflows/ci.yml/badge.svg)](https://github.com/Yorkel/pisa-specification-sensitivity/actions/workflows/ci.yml)

How far does a conclusion about educational attainment move across modelling
choices that are all individually defensible?

This repository does not ask whether low mathematics performance in PISA 2022 UK
data can be predicted. That question is settled and uninteresting. It asks how
much the answer a researcher would report depends on decisions that are rarely
reported at all: which plausible value to analyse, whether to apply the survey
weights, which model family to fit, whether to model a score or a threshold, and
where to place the threshold.

The deliverable is a range, not a point estimate.

## The specification grid

Five axes are crossed over one fixed research question and one fixed dataset.
The threshold axis applies only to the binary target, so the grid is not a full
cross: 36 binary cells and 12 continuous cells, 48 in total. No sampling is
required to stay inside the intended budget.

| Axis | Levels | Rationale |
|---|---|---|
| Plausible value handling | `pv1_only`, `pooled_rubin` | PISA publishes ten plausible values per domain. Analysing one treats a posterior draw as a measurement. Pooling combines all ten under Rubin's rules. |
| Survey weights | `weighted`, `unweighted` | `W_FSTUWT` makes estimates population-representative. Omitting it is common practice in applied machine learning work on PISA. |
| Model family | `pruned_tree`, `random_forest`, `gradient_boosting` | Increasing flexibility, from a single pruned tree to a histogram gradient booster. |
| Target formulation | `binary`, `continuous` | Below-threshold classification against direct regression on the score. |
| Threshold | `level_1a` (357.77), `level_2` (420.07), `level_3` (482.38) | The OECD minimum-proficiency cut plus the adjacent published band boundaries. |

Hyperparameters are held fixed within each model family across the whole grid.
Allowing tuning effort to vary would confound the model-family axis with search
budget.

## Headline result

Across 48 specifications fitted to 12,972 UK students and 35 features:

**The rank of socioeconomic status is not stable, and the threshold choice moves
it most.**

| Quantity | Result |
|---|---:|
| Rank of ESCS, best case | 2nd of 35 |
| Rank of ESCS, worst case | 35th of 35 |
| Median rank of ESCS | 6.5 |
| Specifications placing ESCS first | 0 of 48 |
| Specifications placing ESCS in the top three | 11 of 48 |

![Rank of ESCS across the grid](figures/escs_rank.png)

The distribution is bimodal. ESCS clusters between third and eighth in most
specifications, then collapses to last place in eight of them. It is never the
leading predictor under permutation importance in any cell of the grid.

Movement is attributed by comparing only cells that are identical except on the
axis in question, and averaging the within-set range. Different axes drive
different conclusions:

| Axis | Mean movement in ESCS rank | Mean movement in performance |
|---|---:|---:|
| Threshold | 21.25 places | 0.020 |
| Model family | 11.56 places | 0.083 |
| Plausible value handling | 5.33 places | 0.008 |
| Survey weights | 2.75 places | 0.037 |

Where the low-performance line is drawn moves the socioeconomic ranking by 21
places on average, and by 33 places at worst, while barely moving predictive
performance. Model family does the reverse: it dominates performance and matters
less for the ranking. A study reporting one specification reports one draw from
this distribution.

**Performance itself spans a wide band.**

| Target | Metric | Range across the grid | Median |
|---|---|---:|---:|
| Binary | Weighted AUC | 0.696 to 0.818 | 0.786 |
| Continuous | Weighted R² | 0.235 to 0.433 | 0.348 |

The continuous R² nearly doubles from one end of the grid to the other. Both
figures are defensible; neither is the answer.

**Analysing one plausible value understates uncertainty.** In 23 of 24 matched
comparisons, pooling all ten widened the confidence interval. The median interval
was 1.19 times wider once between-imputation variance was restored, and 1.58
times wider at the extreme. Among pooled cells, between-imputation variance was a
median 25.8% of total variance and reached 55.0%. A single-plausible-value
analysis discards that component entirely.

**One predictor is stable.** Mathematics self-efficacy (`MATHEFF`) appears in the
top five of all 48 specifications. No other feature does. The grid produced 28
distinct top-five sets, with a mean pairwise Jaccard similarity of 0.535, so
roughly half the reported leading predictors change with the specification.

![Specification curve](figures/specification_curve.png)

## Reproducing

The PISA data is not distributed here. Acquire it first:

    python scripts/download_data.py

That prints the OECD source, the file to download, and the filter that produces
the UK extract. Verify a prepared extract with:

    python scripts/download_data.py --verify data/uk_pisa_2022.csv

Then install and run the grid:

    pip install -e '.[test]'
    pisa-specsens --data data/uk_pisa_2022.csv --out results/v1

The command writes `grid_results.csv`, `grid_results.json` and `summary.json`
into the named directory, and refuses to write into a directory that already
holds files. The output behind every number reported above is published under
[results/v1](results/v1), so the figures and the headline table can be checked
without obtaining the PISA data. Results directories name a version; overwriting one silently would
make a published figure impossible to trace to the run that produced it.

Figures are regenerated from a completed results directory:

    pip install -e '.[figures]'
    jupyter nbconvert --to notebook --execute notebooks/figures.ipynb

Run the test suite, which is offline and needs no PISA data:

    pytest

The full grid took 6.1 minutes on an Apple silicon laptop with parallel fitting
enabled. Pooled cells fit ten models each, so 48 cells are 264 fits.

## Repository structure

    src/pisa_specsens/
        config.py           specification axes, proficiency cuts, constants
        data.py             column-restricted loading of the UK extract
        preprocessing.py    feature selection, splitting, leakage-free imputation
        models.py           estimators and permutation importance
        pooling.py          Rubin's rules and bootstrap variance
        grid.py             execution of one cell and of the whole grid
        aggregate.py        rank stability and axis attribution
        cli.py              entry point with the overwrite guard

    tests/                  offline tests on small synthetic frames
    scripts/                data acquisition and verification
    notebooks/figures.ipynb figure production only, no analysis

Training and analysis call the same preprocessing functions. Two specifications
differ only by their declared axes, never by an incidental difference in data
handling.

## Method notes

Features are the 33 standard PISA derived indices rather than raw questionnaire
items, which keeps the feature set documented and comparable to other PISA work.
Two indices exceeding 30% missingness (`LEARRES`, `SDLEFF`) are dropped, because
the PISA questionnaire rotates forms and those items were not put to every
student. Gender and UK nation are one-hot encoded, giving 35 columns.

Median imputation is fitted on the training partition and applied unchanged to
validation and test, so no information from held-out data reaches a model.

Importance is measured by permutation on the held-out partition, not by the
built-in impurity importance. Trees, forests and boosters compute impurity
importance by different internal definitions, so those numbers are not comparable
across the model-family axis. Permutation importance asks the same question of
every model: how much does performance fall when this feature is shuffled.

The binary target marks students below a published OECD proficiency cut. It is
not a median split of the score distribution.

## Limitations

Importance rankings are not causal claims. A feature ranking highly means the
model leans on it, not that changing it would change attainment. Mathematics
self-efficacy and mathematics attainment plainly reinforce each other, and
nothing here separates the directions.

This is one country and one cycle. UK PISA 2022, 12,972 students after
restricting to cases with a final student weight. Nothing here establishes that
the same instability appears in other countries or other years, and the ranking
results in particular should be expected to differ where the questionnaire
composition differs.

Confidence intervals come from bootstrapping the evaluation partition with the
fitted model held fixed. They capture evaluation uncertainty and, for pooled
cells, between-imputation uncertainty. They do not capture uncertainty from
refitting the model, so they are narrower than a full accounting would give.

PISA supplies 80 balanced repeated replication weights for correct standard
errors under its complex sampling design. This study uses the final student
weight but not the replicate weights, so the standard errors are not the ones the
OECD analytical manual prescribes.

Permutation importance is unreliable when features are strongly correlated,
because shuffling one leaves a correlated substitute available to the model.
Several indices here are correlated by construction, `ESCS` and `HOMEPOS` among
them, and part of the ranking instability reported above is likely attributable
to that rather than to the specification axes alone.

The grid holds hyperparameters fixed. A study varying tuning budget as a sixth
axis would likely find further movement.

## Deliberately not included

No causal identification strategy. No attempt to argue that any feature drives
attainment.

No hyperparameter search. Tuning is held fixed so that it cannot be confounded
with the specification axes.

No country comparison, no trend across cycles, and no school-level or
teacher-level modelling. The PISA school and teacher files are not read.

No PISA data. The extract is roughly 53MB and is not redistributed here.

No recommendation about which specification is correct. The point of the grid is
that several are defensible and they disagree.

## Licence

MIT. See [LICENSE](LICENSE).
