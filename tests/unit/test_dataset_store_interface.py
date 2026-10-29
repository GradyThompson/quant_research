"""Tests for the abstract DatasetStore interface."""

import pytest

from quant_research.data.storage import DatasetStore


def test_dataset_store_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DatasetStore()


def test_dataset_store_defines_required_operations():
    required_methods = {
        "write",
        "read",
        "get_metadata",
        "list_versions",
        "exists",
    }

    assert required_methods.issubset(DatasetStore.__abstractmethods__)
