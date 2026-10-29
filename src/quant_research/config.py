from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from quant_research.paths import CONFIG_DIR, resolve_project_path
from quant_research.exceptions import ConfigurationError


class Environment(StrEnum):
    DEVELOPMENT = "development"
    RESEARCH = "research"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class BacktestingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_capital: float = Field(gt=0)
    commission_bps: float = Field(ge=0)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: Environment
    data: DataConfig
    logging: LoggingConfig
    backtesting: BacktestingConfig
    execution: ExecutionConfig


def load_config(environment: str = "development") -> AppConfig:
    """Load and validate an environment configuration."""

    try:
        selected_environment = Environment(environment)
    except ValueError as exc:
        valid = [item.value for item in Environment]

        raise ConfigurationError(
            f"Invalid environment: {environment}. Expected one of: {valid}"
        ) from exc

    config_path = CONFIG_DIR / f"{selected_environment.value}.yaml"

    try:
        with config_path.open(encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)

    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {config_path}") from exc

    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration file: {config_path}") from exc

    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration file: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigurationError(f"Invalid configuration structure: {config_path}")

    try:
        config = AppConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ConfigurationError(f"Configuration validation failed: {config_path}") from exc

    if config.environment != selected_environment:
        raise ConfigurationError(
            f"Environment mismatch in {config_path}: "
            f"expected {selected_environment.value}, "
            f"received {config.environment.value}"
        )

    config.data.path = resolve_project_path(config.data.path)

    return config
