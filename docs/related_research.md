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

## Importance under correlated features

**Hooker, G., Mentch, L. and Zhou, S. (2021). Unrestricted permutation forces
extrapolation: variable importance requires at least one more model, or there is
no free variable importance. *Statistics and Computing* 31(6), 82.
[doi:10.1007/s11222-021-10057-z](https://doi.org/10.1007/s11222-021-10057-z)**

Permuting a feature independently of its correlates evaluates the model in
regions of the feature space where no data exists, which distorts the resulting
importance. The paper recommends measuring performance change after muting a
feature's effect rather than permuting it in isolation.

This bears directly on the headline result. ESCS correlates 0.807 with HISEI,
0.753 with HOMEPOS and 0.739 with PAREDINT, because OECD constructs ESCS from
those indices. Single-feature permutation cannot separate them: shuffling ESCS
leaves three substitutes in place. The grouped permutation reported in the README
is the response to this paper, and it is a partial one. A full response would fit
an additional model per feature, as the title of the paper indicates.
