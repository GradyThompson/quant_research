from typing import Any

import yaml

from quant_research.paths import CONFIG_DIR

VALID_ENVIRONMENTS = {"development", "research"}


def load_config(environment: str = "development") -> dict[str, Any]:
    """Load configuration for the specified environment."""

    if environment not in VALID_ENVIRONMENTS:
        raise ValueError(
            f"Invalid environment: {environment}. Expected one of: {sorted(VALID_ENVIRONMENTS)}"
        )

    config_path = CONFIG_DIR / f"{environment}.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration: {config_path}")

    if config.get("environment") != environment:
        raise ValueError(f"Environment mismatch in {config_path}")

    return config
