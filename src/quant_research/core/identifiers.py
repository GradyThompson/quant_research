"""Stable identifiers used throughout the research platform."""

import re
from dataclasses import dataclass
from uuid import UUID

_SECURITY_ID_PATTERN = re.compile(r"^SEC-[0-9]{6,}$")

_DATASET_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


@dataclass(frozen=True, slots=True)
class SecurityId:
    """Stable internal identifier for a security."""

    value: str

    def __post_init__(self) -> None:
        if not _SECURITY_ID_PATTERN.fullmatch(self.value):
            raise ValueError("SecurityId must match 'SEC-' followed by at least six digits.")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class DatasetId:
    """Logical identifier for a dataset."""

    value: str

    def __post_init__(self) -> None:
        if not _DATASET_ID_PATTERN.fullmatch(self.value):
            raise ValueError(
                "DatasetId must contain at least two lowercase dot-separated components."
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """Identifier for a particular immutable dataset version."""

    value: UUID

    def __str__(self) -> str:
        return str(self.value)
