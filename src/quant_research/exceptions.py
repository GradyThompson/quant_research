"""Application-specific exceptions for quant_research."""


class QuantResearchError(Exception):
    """Base class for quant_research application errors."""


class ConfigurationError(QuantResearchError):
    """Raised when application configuration is invalid."""


class DataError(QuantResearchError):
    """Base class for data-related errors."""


class DataValidationError(DataError):
    """Raised when data violates a schema or validation rule."""


class DatasetNotFoundError(DataError):
    """Raised when a requested dataset cannot be found."""


class DatasetIntegrityError(DataError):
    """Raised when stored data is corrupted or inconsistent."""


class ResearchError(QuantResearchError):
    """Raised when a research operation fails."""
