import pytest

from quant_research.config import load_config


def test_load_development_config():
    config = load_config("development")

    assert config["environment"] == "development"
    assert config["execution"]["enabled"] is False


def test_load_research_config():
    config = load_config("research")

    assert config["environment"] == "research"
    assert config["execution"]["enabled"] is False


def test_invalid_environment():
    with pytest.raises(ValueError):
        load_config("invalid")
