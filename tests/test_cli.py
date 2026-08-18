import pytest

from pisa_specsens.cli import OutputDirectoryNotEmpty, build_parser, main, require_empty_output


def test_empty_directory_is_accepted(tmp_path):
    target = tmp_path / "v1"
    assert require_empty_output(target) == target
    assert target.exists()


def test_non_empty_directory_is_refused(tmp_path):
    """Results directories name a version and must never be overwritten silently."""
    target = tmp_path / "v1"
    target.mkdir()
    (target / "grid_results.csv").write_text("existing")
    with pytest.raises(OutputDirectoryNotEmpty):
        require_empty_output(target)


def test_cli_returns_error_code_for_non_empty_output(tmp_path, capsys):
    target = tmp_path / "v1"
    target.mkdir()
    (target / "summary.json").write_text("{}")
    code = main(["--data", str(tmp_path / "absent.csv"), "--out", str(target)])
    assert code == 2


def test_cli_returns_error_code_for_missing_data(tmp_path):
    code = main(["--data", str(tmp_path / "absent.csv"), "--out", str(tmp_path / "fresh")])
    assert code == 3


def test_parser_requires_data_and_out():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
