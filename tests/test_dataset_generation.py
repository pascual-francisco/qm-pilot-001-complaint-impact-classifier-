"""
test_dataset_generation.py

REQ Coverage:
- REQ-001
- REQ-011
- REQ-012

Verification:
- OQ-001
- OQ-002
"""

import pandas as pd
import pytest

from complaint_classifier.generate_dataset import (
    ALLOWED_IMPACTS,
    generate_dataset,
    validate_dataset,
)


def test_req_001_dataset_contains_minimum_records() -> None:
    """
    REQ-001 / OQ-001

    Verify that the generated dataset contains
    at least 500 records.
    """
    dataset = generate_dataset()

    assert len(dataset) >= 500


def test_req_011_rejects_dataset_smaller_than_500() -> None:
    """
    REQ-011 / OQ-001

    Verify that datasets smaller than the minimum
    allowed size are rejected.
    """
    with pytest.raises(ValueError):
        generate_dataset(
            record_count=499,
        )


def test_req_012_required_columns_exist() -> None:
    """
    REQ-012 / OQ-002

    Verify schema completeness.
    """
    dataset = generate_dataset()

    expected_columns = {
        "complaint_id",
        "product",
        "category",
        "initial_severity",
        "days_open",
        "repeat_event",
        "vip_customer",
        "impact",
    }

    assert expected_columns.issubset(
        set(dataset.columns)
    )


def test_req_012_unique_complaint_ids() -> None:
    """
    REQ-012 / OQ-002

    Verify complaint identifiers are unique.
    """
    dataset = generate_dataset()

    assert dataset["complaint_id"].is_unique


def test_req_012_valid_impact_values() -> None:
    """
    REQ-012 / OQ-002

    Verify allowed target values only.
    """
    dataset = generate_dataset()

    invalid_values = (
        set(dataset["impact"])
        - set(ALLOWED_IMPACTS)
    )

    assert len(invalid_values) == 0


def test_req_012_validation_fails_when_column_missing() -> None:
    """
    REQ-012 / OQ-002

    Verify schema validation fails when
    a required column is removed.
    """
    dataset = generate_dataset()

    dataset = dataset.drop(
        columns=["impact"]
    )

    with pytest.raises(ValueError):
        validate_dataset(dataset)


def test_req_012_validation_fails_for_duplicate_ids() -> None:
    """
    REQ-012 / OQ-002

    Verify duplicate complaint IDs are detected.
    """
    dataset = generate_dataset()

    duplicate_row = dataset.iloc[[0]]

    dataset = pd.concat(
        [
            dataset,
            duplicate_row,
        ],
        ignore_index=True,
    )

    dataset.loc[
        len(dataset) - 1,
        "complaint_id",
    ] = dataset.loc[
        0,
        "complaint_id",
    ]

    with pytest.raises(ValueError):
        validate_dataset(dataset)


def test_dataset_generation_is_reproducible() -> None:
    """
    REQ-006 support test

    Verify reproducibility using the same seed.
    """
    dataset_a = generate_dataset(
        random_seed=42,
    )

    dataset_b = generate_dataset(
        random_seed=42,
    )

    pd.testing.assert_frame_equal(
        dataset_a,
        dataset_b,
    )
