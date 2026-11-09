"""
split_dataset.py

REQ Coverage:
- REQ-013: Perform controlled train, validation, and test splitting.
- REQ-006: Document reproducibility parameters.

Risk Coverage:
- RISK-003: Data leakage between train and test datasets.

Verification:
- OQ-003
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_TRAIN_SIZE: Final[float] = 0.70
DEFAULT_VALIDATION_SIZE: Final[float] = 0.15
DEFAULT_TEST_SIZE: Final[float] = 0.15


def split_dataset(
    dataset: pd.DataFrame,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: Source dataset.
        random_seed: Seed for reproducible splitting.

    Returns:
        Tuple containing:
            train_df,
            validation_df,
            test_df

    Raises:
        ValueError: If dataset is invalid.
    """
    validate_split_inputs(dataset)

    train_df, temp_df = train_test_split(
        dataset,
        test_size=(DEFAULT_VALIDATION_SIZE + DEFAULT_TEST_SIZE),
        random_state=random_seed,
        stratify=dataset["impact"],
    )

    validation_ratio = (
        DEFAULT_VALIDATION_SIZE
        / (DEFAULT_VALIDATION_SIZE + DEFAULT_TEST_SIZE)
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - validation_ratio),
        random_state=random_seed,
        stratify=temp_df["impact"],
    )

    validate_split_outputs(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        original_rows=len(dataset),
    )

    LOGGER.info(
        (
            "Dataset split completed. "
            "train=%s validation=%s test=%s seed=%s"
        ),
        len(train_df),
        len(validation_df),
        len(test_df),
        random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def validate_split_inputs(dataset: pd.DataFrame) -> None:
    """
    Validate dataset before splitting.

    Args:
        dataset: Dataset to validate.

    Raises:
        ValueError: If validation fails.
    """
    if dataset.empty:
        raise ValueError(
            "REQ-013 violation: Dataset cannot be empty."
        )

    if "impact" not in dataset.columns:
        raise ValueError(
            "REQ-013 violation: Missing target column 'impact'."
        )


def validate_split_outputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_rows: int,
) -> None:
    """
    Verify split integrity and leakage prevention.

    Args:
        train_df: Training dataset.
        validation_df: Validation dataset.
        test_df: Test dataset.
        original_rows: Original row count.

    Raises:
        ValueError: If integrity checks fail.
    """
    train_ids = set(train_df["complaint_id"])
    validation_ids = set(validation_df["complaint_id"])
    test_ids = set(test_df["complaint_id"])

    if train_ids.intersection(validation_ids):
        raise ValueError(
            "REQ-013 violation: Train/Validation leakage detected."
        )

    if train_ids.intersection(test_ids):
        raise ValueError(
            "REQ-013 violation: Train/Test leakage detected."
        )

    if validation_ids.intersection(test_ids):
        raise ValueError(
            "REQ-013 violation: Validation/Test leakage detected."
        )

    total_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_rows != original_rows:
        raise ValueError(
            (
                "REQ-013 violation: "
                "Split row counts do not reconcile."
            )
        )


def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save split datasets.

    Args:
        train_df: Training data.
        validation_df: Validation data.
        test_df: Test data.
    """
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/validation").mkdir(parents=True, exist_ok=True)
    Path("data/test").mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        "data/train/complaints_train.csv",
        index=False,
    )

    validation_df.to_csv(
        "data/validation/complaints_validation.csv",
        index=False,
    )

    test_df.to_csv(
        "data/test/complaints_test.csv",
        index=False,
    )

    LOGGER.info("Split datasets saved successfully.")


def load_dataset(
    input_file: str,
) -> pd.DataFrame:
    """
    Load source dataset.

    Args:
        input_file: Dataset path.

    Returns:
        Loaded DataFrame.
    """
    return pd.read_csv(input_file)


def main() -> None:
    """
    Execute dataset splitting workflow.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dataset = load_dataset(
        "data/raw/complaints_dataset_v1.csv"
    )

    train_df, validation_df, test_df = split_dataset(
        dataset=dataset,
        random_seed=DEFAULT_RANDOM_SEED,
    )

    save_splits(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )


if __name__ == "__main__":
    main()
