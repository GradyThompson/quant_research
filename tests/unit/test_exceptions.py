import pytest

from quant_research.exceptions import (
    ConfigurationError,
    DataError,
    DatasetIntegrityError,
    DatasetNotFoundError,
    DataValidationError,
    QuantResearchError,
    ResearchError,
)


def test_configuration_error_inheritance():
    assert issubclass(ConfigurationError, QuantResearchError)


def test_data_error_inheritance():
    assert issubclass(DataError, QuantResearchError)


def test_data_validation_error_inheritance():
    assert issubclass(DataValidationError, DataError)


def test_dataset_not_found_error_inheritance():
    assert issubclass(DatasetNotFoundError, DataError)


def test_dataset_integrity_error_inheritance():
    assert issubclass(DatasetIntegrityError, DataError)


def test_research_error_inheritance():
    assert issubclass(ResearchError, QuantResearchError)


def test_specific_error_can_be_caught_as_base_error():
    with pytest.raises(QuantResearchError):
        raise ConfigurationError("Invalid configuration")


def test_data_error_can_catch_specialized_errors():
    with pytest.raises(DataError):
        raise DatasetNotFoundError("Missing OHLCV dataset")


def test_exception_preserves_message():
    error = DatasetIntegrityError("Dataset checksum mismatch")

    assert str(error) == "Dataset checksum mismatch"
