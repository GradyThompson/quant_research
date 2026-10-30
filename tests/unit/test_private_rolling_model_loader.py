"""Regression tests for loading local dataclasses and sibling model modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script(path: Path) -> object:
    spec = importlib.util.spec_from_file_location("test_rolling_cli_loader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_load_supports_dataclasses_and_sibling_imports(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    directory.mkdir()
    (directory / "features.py").write_text("VALUE = 42\n", encoding="utf-8")
    (directory / "rolling_model.py").write_text(
        "from dataclasses import dataclass\n"
        "from features import VALUE\n"
        "@dataclass\n"
        "class PrivateRollingBarrierModel:\n"
        "    value: int = VALUE\n"
        "    def initialize(self, data): pass\n"
        "    def update(self, new_bars, *, decision_at): pass\n"
        "    def forecast(self, **kwargs): pass\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_historical_rolling_barrier.py"
    module = _load_script(script)
    old_path = list(sys.path)
    model = module.load_private_rolling_model(directory / "rolling_model.py")  # type: ignore[attr-defined]
    assert model.value == 42
    assert sys.path == old_path


def test_missing_private_model(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_historical_rolling_barrier.py"
    module = _load_script(script)
    with pytest.raises(FileNotFoundError):
        module.load_private_rolling_model(tmp_path / "missing.py")  # type: ignore[attr-defined]
