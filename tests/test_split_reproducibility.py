"""
test_split_reproducibility.py

REQ Coverage:
- REQ-013

Risk Coverage:
- RISK-003

Verification:
- OQ-003
"""

import pandas as pd
import pytest

from complaint_classifier.generate_dataset import generate_dataset
from complaint_classifier.split_dataset import (
    split_dataset,
    validate_split_inputs,
)


def test_req_013_split_reproducibility() -> None:
    """
    REQ-013 / OQ-003

    Verify identical splits using the same seed.
    """
    dataset = generate_dataset()

    train_a, validation_a, test_a = split_dataset(
        dataset=dataset,
        random_seed=42,
    )

    train_b, validation_b, test_b = split_dataset(
        dataset=dataset,
        random_seed=42,
    )

    pd.testing.assert_frame_equal(
        train_a,
        train_b,
    )

    pd.testing.assert_frame_equal(
        validation_a,
        validation_b,
    )

    pd.testing.assert_frame_equal(
        test_a,
        test_b,
    )


def test_req_013_no_train_validation_leakage() -> None:
    """
    REQ-013 / OQ-003

    Verify train and validation datasets
    do not share complaint IDs.
    """
    dataset = generate_dataset()

    train_df, validation_df, _ = split_dataset(
        dataset=dataset,
        random_seed=42,
    )

    overlap = set(
        train_df["complaint_id"]
    ).intersection(
        set(validation_df["complaint_id"])
    )

    assert len(overlap) == 0


def test_req_013_no_train_test_leakage() -> None:
    """
    REQ-013 / OQ-003

    Verify train and test datasets
    do not share complaint IDs.
    """
    dataset = generate_dataset()

    train_df, _, test_df = split_dataset(
        dataset=dataset,
        random_seed=42,
    )

    overlap = set(
        train_df["complaint_id"]
    ).intersection(
        set(test_df["complaint_id"])
    )

    assert len(overlap) == 0


def test_req_013_no_validation_test_leakage() -> None:
    """
    REQ-013 / OQ-003

    Verify validation and test datasets
    do not share complaint IDs.
    """
    dataset = generate_dataset()

    _, validation_df, test_df = split_dataset(
        dataset=dataset,
        random_seed=42,
    )

    overlap = set(
        validation_df["complaint_id"]
    ).intersection(
        set(test_df["complaint_id"])
    )

    assert len(overlap) == 0


def test_req_013_row_reconciliation() -> None:
    """
    REQ-013 / OQ-003

    Verify rows reconcile after split.
    """
    dataset = generate_dataset()

    train_df, validation_df, test_df = split_dataset(
        dataset=dataset,
    )

    output_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    assert output_rows == len(dataset)


def test_req_013_reject_empty_dataset() -> None:
    """
    REQ-013 / OQ-003

    Verify empty datasets are rejected.
    """
    empty_dataset = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_split_inputs(
            empty_dataset,
        )


def test_req_013_reject_missing_target_column() -> None:
    """
    REQ-013 / OQ-003

    Verify datasets missing the
    target column are rejected.
    """
    dataset = generate_dataset()

    dataset = dataset.drop(
        columns=["impact"]
    )

    with pytest.raises(ValueError):
        validate_split_inputs(
            dataset,
        )
