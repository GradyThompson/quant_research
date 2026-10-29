from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import quant_research.config as config_module
from quant_research.config import (
    AppConfig,
    Environment,
    load_config,
)
from quant_research.paths import PROJECT_ROOT


def test_load_development_config():
    config = load_config("development")

    assert isinstance(config, AppConfig)
    assert config.environment == Environment.DEVELOPMENT
    assert config.data.path == PROJECT_ROOT / "data" / "dev"
    assert config.data.path.is_absolute()


def test_load_research_config():
    config = load_config("research")

    assert config.environment == Environment.RESEARCH
    assert config.data.path == PROJECT_ROOT / "data" / "research"
    assert config.data.path.is_absolute()


def test_default_environment():
    config = load_config()

    assert config.environment == Environment.DEVELOPMENT


def test_invalid_environment():
    with pytest.raises(ValueError, match="Invalid environment"):
        load_config("invalid")


def test_negative_initial_capital():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "environment": "development",
                "data": {"path": "data/dev"},
                "logging": {"level": "INFO"},
                "backtesting": {
                    "initial_capital": -100000,
                    "commission_bps": 1.0,
                },
                "execution": {"enabled": False},
            }
        )


def test_negative_commission():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "environment": "development",
                "data": {"path": "data/dev"},
                "logging": {"level": "INFO"},
                "backtesting": {
                    "initial_capital": 100000,
                    "commission_bps": -1.0,
                },
                "execution": {"enabled": False},
            }
        )


def test_invalid_execution_enabled():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "environment": "development",
                "data": {"path": "data/dev"},
                "logging": {"level": "INFO"},
                "backtesting": {
                    "initial_capital": 100000,
                    "commission_bps": 1.0,
                },
                "execution": {"enabled": "sometimes"},
            }
        )


def test_unknown_configuration_field():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "environment": "development",
                "data": {"path": "data/dev"},
                "logging": {"level": "INFO"},
                "backtesting": {
                    "initial_capital": 100000,
                    "commission_bps": 1.0,
                    "unknown_field": 123,
                },
                "execution": {"enabled": False},
            }
        )


def test_environment_mismatch(tmp_path, monkeypatch):
    config_path = tmp_path / "development.yaml"

    config_path.write_text(
        yaml.safe_dump(
            {
                "environment": "research",
                "data": {"path": "data/dev"},
                "logging": {"level": "INFO"},
                "backtesting": {
                    "initial_capital": 100000,
                    "commission_bps": 1.0,
                },
                "execution": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    with pytest.raises(ValueError, match="Environment mismatch"):
        load_config("development")


def test_config_path_independent_of_working_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    config = load_config("research")

    expected = PROJECT_ROOT / "data" / "research"

    assert config.data.path == expected
