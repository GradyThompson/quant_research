from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
TESTS_DIR = PROJECT_ROOT / "tests"


# ---------------------------------------------------------
# Path resolution
# ---------------------------------------------------------


def resolve_project_path(
    path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a path relative to the project root.

    Absolute paths are preserved.

    Relative paths are resolved against the project root,
    not the current working directory.

    No directories are created.
    """

    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()
