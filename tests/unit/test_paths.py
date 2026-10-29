from pathlib import Path

from quant_research.paths import PROJECT_ROOT, resolve_project_path


def test_project_root_is_absolute():
    assert PROJECT_ROOT.is_absolute()


def test_project_root_contains_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_resolve_relative_path(tmp_path):
    result = resolve_project_path(
        "data/research",
        project_root=tmp_path,
    )

    assert result == tmp_path / "data" / "research"
    assert result.is_absolute()


def test_resolve_absolute_path(tmp_path):
    absolute_path = tmp_path / "external" / "market_data"

    result = resolve_project_path(
        absolute_path,
        project_root=tmp_path / "other",
    )

    assert result == absolute_path


def test_resolve_path_does_not_create_directories(tmp_path):
    result = resolve_project_path(
        "data/new",
        project_root=tmp_path,
    )

    assert not result.exists()


def test_resolve_path_is_independent_of_working_directory(
    tmp_path,
    monkeypatch,
):
    project_root = tmp_path / "project"
    other_directory = tmp_path / "other"

    project_root.mkdir()
    other_directory.mkdir()

    monkeypatch.chdir(other_directory)

    result = resolve_project_path(
        "data/research",
        project_root=project_root,
    )

    assert result == project_root / "data" / "research"


def test_resolve_path_accepts_path_objects(tmp_path):
    result = resolve_project_path(
        Path("data/research"),
        project_root=tmp_path,
    )

    assert result == tmp_path / "data" / "research"
