"""Fixed constants and the specification grid definition.

Values here determine what the study claims. They are declared in one place so
that a reader can audit every modelling choice without reading the fitting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

# OECD PISA 2022 mathematics proficiency band lower bounds, in PISA score points.
# Source: OECD (2023), PISA 2022 Results Volume I, Annex A1.
PROFICIENCY_CUTS = {
    "level_1a": 357.77,
    "level_2": 420.07,  # the OECD "minimum proficiency" threshold
    "level_3": 482.38,
}

# The ten plausible values for mathematics.
PV_COLUMNS = [f"PV{i}MATH" for i in range(1, 11)]

# Final student weight.
WEIGHT_COLUMN = "W_FSTUWT"

# PISA supplies 80 Fay balanced repeated replication weights. The OECD analytical
# manual prescribes these for standard errors under the complex sampling design.
REPLICATE_WEIGHT_COLUMNS = [f"W_FSTURWT{i}" for i in range(1, 81)]
FAY_COEFFICIENT = 0.5

# Standard PISA derived indices (weighted likelihood estimates). Using the
# published indices rather than raw questionnaire items keeps the feature set
# documented and comparable to other PISA work. Indices with more than 30%
# missingness in the UK file (LEARRES, SDLEFF) are excluded before modelling.
CANDIDATE_INDICES = [
    "ESCS", "HOMEPOS", "HISEI", "PAREDINT", "ICTRES", "BELONG", "ANXMAT",
    "MATHEFF", "MATHPERS", "TEACHSUP", "DISCLIM", "FAMSUP", "BULLIED",
    "FEELSAFE", "EXPOFA", "EXPO21ST", "COGACRCO", "CURIOAGR", "GROSAGR",
    "STRESAGR", "SDLEFF", "RELATST", "SCHRISK", "LEARRES", "FAMCON",
    "ICTHOME", "ICTSCH", "WORKPAY", "WORKHOME", "SKIPPING", "TARDYSD",
    "STUDYHMW", "MATHPREF",
]

CATEGORICAL_COLUMNS = ["ST004D01T", "REGION"]  # gender, UK nation

# Columns dropped when missingness exceeds this share of rows.
MISSINGNESS_DROP_THRESHOLD = 0.30

# Data splitting.
TEST_SIZE = 0.20
VAL_SIZE = 0.20
RANDOM_SEED = 20220101

# Bootstrap resamples used for confidence intervals on performance metrics.
BOOTSTRAP_RESAMPLES = 400

# Repeats used when measuring permutation importance on the held-out partition.
PERMUTATION_REPEATS = 5

# Features that OECD combines to construct ESCS, plus the home ICT resource index
# that overlaps with home possessions. Permuting any one of these alone leaves the
# others available as substitutes, so the group is also permuted jointly.
BACKGROUND_BLOCK = ["ESCS", "HISEI", "HOMEPOS", "PAREDINT", "ICTRES"]

# Mathematics disposition items that correlate with each other. Included so that
# the stability of MATHEFF is subjected to the same test as ESCS.
DISPOSITION_BLOCK = ["MATHEFF", "FAMCON"]

FEATURE_BLOCKS = {
    "background": BACKGROUND_BLOCK,
    "disposition": DISPOSITION_BLOCK,
}

# The feature whose rank stability is the headline conclusion of the study.
FOCAL_FEATURE = "ESCS"

# Number of top predictors recorded per specification.
TOP_K = 5

# Specification axes.
PV_HANDLING = ["pv1_only", "pooled_rubin"]
WEIGHTING = ["weighted", "unweighted"]
MODEL_FAMILY = ["pruned_tree", "random_forest", "gradient_boosting"]
TARGET_FORM = ["binary", "continuous"]
THRESHOLD = ["level_1a", "level_2", "level_3"]


@dataclass(frozen=True)
class Specification:
    """One cell of the specification grid."""

    pv_handling: str
    weighting: str
    model_family: str
    target_form: str
    threshold: str | None

    @property
    def cell_id(self) -> str:
        parts = [self.pv_handling, self.weighting, self.model_family, self.target_form]
        if self.threshold is not None:
            parts.append(self.threshold)
        return "__".join(parts)

    def as_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "pv_handling": self.pv_handling,
            "weighting": self.weighting,
            "model_family": self.model_family,
            "target_form": self.target_form,
            "threshold": self.threshold,
        }


def build_grid() -> list[Specification]:
    """Enumerate the specification grid.

    The threshold axis applies only to the binary target formulation, so the
    grid is not a full cross of all five axes. Binary contributes
    2 x 2 x 3 x 3 = 36 cells and continuous contributes 2 x 2 x 3 = 12,
    giving 48 cells in total.
    """
    specs: list[Specification] = []
    for pv, wt, model in product(PV_HANDLING, WEIGHTING, MODEL_FAMILY):
        for cut in THRESHOLD:
            specs.append(Specification(pv, wt, model, "binary", cut))
        specs.append(Specification(pv, wt, model, "continuous", None))
    return specs
