# Related research

What each paper establishes, and how it bears on the design of this study. Three
separate literatures are relevant, and they answer different questions. Conflating
them is how a study of this kind ends up overclaiming.

## Specification multiplicity

**Simonsohn, U., Simmons, J. P. and Nelson, L. D. (2020). Specification curve
analysis. *Nature Human Behaviour* 4(11), 1208-1214.
[doi:10.1038/s41562-020-0912-z](https://doi.org/10.1038/s41562-020-0912-z)**

Sets out the method this repository implements. A researcher faces many
specifications that are all defensible, statistically valid and non-redundant.
Reporting one of them reports a single draw from a distribution the reader never
sees, and the choice of draw is likely to favour the author's narrative. The
paper proposes enumerating the set, displaying the results as a curve, and
conducting inference jointly across it.

The three steps map directly onto this repository: `config.build_grid` enumerates,
`notebooks/figures.ipynb` displays, `aggregate.py` performs the joint summary.
The departure is that Simonsohn et al. apply the method to effect estimates,
whereas the outcome here is a feature importance ranking.

**Steegen, S., Tuerlinckx, F., Gelman, A. and Vanpaemel, W. (2016). Increasing
transparency through a multiverse analysis. *Perspectives on Psychological
Science* 11(5), 702-712.
[doi:10.1177/1745691616658637](https://doi.org/10.1177/1745691616658637)**

The same argument aimed at data processing rather than model specification:
exclusion rules, variable operationalisation and coding choices each generate a
different dataset, and the set of them is a multiverse. Relevant here because the
plausible value and threshold axes are data-construction choices, not model
choices. The target variable is literally different across the threshold axis.

**Robitzsch, A. (2022). Exploring the multiverse of analytical decisions in
scaling educational large-scale assessment data: a specification curve analysis
for PISA 2018 mathematics data. *European Journal of Investigation in Health,
Psychology and Education* 12(7), 731-753.
[doi:10.3390/ejihpe12070054](https://doi.org/10.3390/ejihpe12070054)**

The nearest precedent, and the one an editor in this field will expect to see
cited. Runs a specification curve on PISA 2018 mathematics across five axes: the
functional form of the item response model, country-level differential item
functioning, the treatment of missing item responses, item selection, and test
position effects. Reports that model uncertainty affects country means about as
much as student sampling error does, and affects country standard deviations more
than standard errors do.

Establishes that specification curve analysis on PISA is already done. The
difference is the layer. Robitzsch varies the psychometric scaling that produces
the plausible values, and tracks country distribution parameters. This grid takes
the published plausible values as given and varies decisions a secondary analyst
makes afterwards, tracking a feature importance ranking. The two studies are
adjacent rather than competing: they examine different halves of the same
pipeline.

## Model multiplicity

**Fisher, A., Rudin, C. and Dominici, F. (2019). All models are wrong, but many
are useful: learning a variable's importance by studying an entire class of
prediction models simultaneously. *Journal of Machine Learning Research* 20(177),
1-81. [jmlr.org/papers/v20/18-760](https://jmlr.org/papers/v20/18-760.html)**

The closest prior work, and the reason this repository does not claim novelty for
the idea of varying importance across models. Model class reliance measures the
range of a variable's importance across every well-performing model in a class,
on the argument that a variable important to one good model may be unimportant to
another. This is the Rashomon perspective: many models fit the data about equally
well and disagree about why.

The distinction from this study is the location of the choice. Model class
reliance varies the model while holding the data and the estimand fixed. The grid
here varies choices upstream of the model, including what the outcome variable is.
A method that searches the space of well-performing models cannot report that the
low-performance threshold was placed arbitrarily, because threshold placement is
not a property of the model.

**D'Amour, A., Heller, K., Moldovan, D., Adlam, B. et al. (2022).
Underspecification presents challenges for credibility in modern machine
learning. *Journal of Machine Learning Research* 23(226), 1-61.
[jmlr.org/papers/v23/20-1335](https://www.jmlr.org/papers/v23/20-1335.html)**

Shows that pipelines returning predictors with equivalent held-out performance
produce predictors that behave very differently once deployed. The credibility
problem is not that any single model is wrong but that the pipeline does not
determine which model you get. Supports the framing here that equivalent test
performance across the grid is not evidence that the specifications agree.

**Dong, J. and Rudin, C. (2020). Exploring the cloud of variable importance for
the set of all good models. *Nature Machine Intelligence* 2(12), 810-824.
[doi:10.1038/s42256-020-00264-0](https://doi.org/10.1038/s42256-020-00264-0)**

Extends the Rashomon argument into a variable importance cloud: rather than a
single importance value, the set of good models yields a region of values, and
the shape of that region is the honest report. Directly relevant to the framing
of the first finding here. No located work in this line handles plausible values,
replicate weights or complex sampling, which is where this study sits.

## Measurement in large-scale assessment

**Rutkowski, L., Gonzalez, E., Joncas, M. and von Davier, M. (2010).
International large-scale assessment data: issues in secondary analysis and
reporting. *Educational Researcher* 39(2), 142-151.
[doi:10.3102/0013189X10363170](https://doi.org/10.3102/0013189X10363170)**

Documents that secondary analyses of PISA and similar assessments routinely
mishandle plausible values and sampling weights. This is why the plausible value
axis is a study of a known failure mode rather than a hypothetical one. The study
measures what the mishandling costs on one dataset instead of restating that it
happens.

## Dichotomisation

**Royston, P., Altman, D. G. and Sauerbrei, W. (2006). Dichotomizing continuous
predictors in multiple regression: a bad idea. *Statistics in Medicine* 25(1),
127-141. [doi:10.1002/sim.2331](https://doi.org/10.1002/sim.2331)**

The standard reference for the costs of cutting a continuous variable into
categories: loss of power, residual confounding, and severe bias when the cut
point is chosen from the data. Motivates the threshold axis. The result reported
here is a variation on the theme rather than a contradiction of it: the three
thresholds are published OECD band boundaries rather than data-derived optima, so
the bias Royston et al. warn about does not arise, and yet the choice among them
still moves the importance ranking by 21 places while leaving predictive
performance nearly unchanged.

## Importance under correlated features

**Hooker, G., Mentch, L. and Zhou, S. (2021). Unrestricted permutation forces
extrapolation: variable importance requires at least one more model, or there is
no free variable importance. *Statistics and Computing* 31(6), 82.
[doi:10.1007/s11222-021-10057-z](https://doi.org/10.1007/s11222-021-10057-z)**

Permuting a feature independently of its correlates evaluates the model in
regions of the feature space where no data exists, which distorts the resulting
importance. The paper recommends measuring performance change after muting a
feature's effect rather than permuting it in isolation.

**Gregorutti, B., Michel, B. and Saint-Pierre, P. (2015). Grouped variable
importance with random forests and application to multiple functional data
analysis. *Computational Statistics and Data Analysis* 90, 15-35.
[doi:10.1016/j.csda.2015.04.002](https://doi.org/10.1016/j.csda.2015.04.002)**

Introduces permuting a group of variables jointly and develops the theory for
additive models. The companion paper, Gregorutti et al. (2017,
[doi:10.1007/s11222-016-9646-1](https://doi.org/10.1007/s11222-016-9646-1)),
derives how correlation deflates single-feature permutation importance. This is
the estimator used in the grouped results reported in the README. It is not a
contribution of this repository.

**Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. and Zeileis, A. (2008).
Conditional variable importance for random forests. *BMC Bioinformatics* 9, 307.
[doi:10.1186/1471-2105-9-307](https://doi.org/10.1186/1471-2105-9-307)**

Diagnoses the bias of unconditional single-feature permutation under correlated
predictors and proposes conditional permutation as the remedy. Nicodemus et al.
(2010, [doi:10.1186/1471-2105-11-110](https://doi.org/10.1186/1471-2105-11-110))
documents the behaviour empirically. Conditional permutation is the alternative
this study does not implement.

### What is and is not claimed here

The mechanism is established. ESCS correlates 0.807 with HISEI, 0.753 with
HOMEPOS and 0.739 with PAREDINT because OECD constructs ESCS from those indices,
so single-feature permutation leaves three substitutes in place. That correlated
predictors deflate permutation importance, and that joint permutation of the
correlated group addresses it, is Gregorutti's and Strobl's result.

What this repository contributes on that point is narrower: the demonstration
that the artefact is severe enough in PISA to move ESCS from second to last place
across defensible specifications, and that the block defined by OECD's own
construction rule resolves it completely. The blocks are declared, not
discovered, and the remedy is partial: it removes substitution within a block but
not across blocks, and a full answer requires the additional model that Hooker,
Mentch and Zhou argue is unavoidable.
