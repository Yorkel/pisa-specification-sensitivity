from pisa_specsens.config import PROFICIENCY_CUTS, build_grid


def test_grid_size_is_forty_eight():
    """Threshold applies only to the binary target, so the grid is not a full cross."""
    grid = build_grid()
    assert len(grid) == 48
    assert sum(1 for s in grid if s.target_form == "binary") == 36
    assert sum(1 for s in grid if s.target_form == "continuous") == 12


def test_continuous_cells_carry_no_threshold():
    for spec in build_grid():
        if spec.target_form == "continuous":
            assert spec.threshold is None
        else:
            assert spec.threshold in PROFICIENCY_CUTS


def test_cell_ids_are_unique():
    ids = [s.cell_id for s in build_grid()]
    assert len(ids) == len(set(ids))


def test_level_two_cut_is_the_oecd_value():
    assert PROFICIENCY_CUTS["level_2"] == 420.07
    assert PROFICIENCY_CUTS["level_1a"] < PROFICIENCY_CUTS["level_2"]
    assert PROFICIENCY_CUTS["level_3"] > PROFICIENCY_CUTS["level_2"]
