"""
generate_dataset.py

REQ Coverage:
- REQ-001: Generate a synthetic complaint dataset containing at least 500 records.
- REQ-011: Validate dataset size before use.
- REQ-012: Produce records that conform to the approved schema and allowed values.

Purpose:
Generate a reproducible synthetic complaint dataset for development,
validation, and testing of the Complaint Impact Classifier project.

Data source:
Synthetic only.

Author:
QM-PILOT-001 Complaint Impact Classifier
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_RECORD_COUNT: Final[int] = 500
DEFAULT_RANDOM_SEED: Final[int] = 42

ALLOWED_PRODUCTS: Final[list[str]] = [
    "Neuro",
    "Cardiac",
    "Surgical",
    "Monitoring",
]

ALLOWED_CATEGORIES: Final[list[str]] = [
    "Labeling",
    "Packaging",
    "Functional",
    "Cosmetic",
]

ALLOWED_IMPACTS: Final[list[str]] = [
    "Low",
    "Medium",
    "High",
]


def generate_dataset(
    record_count: int = DEFAULT_RECORD_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic complaint dataset.

    Args:
        record_count: Number of records to generate.
        random_seed: Seed used for reproducibility.

    Returns:
        pd.DataFrame containing synthetic complaint records.

    Raises:
        ValueError: If record_count is less than 500.
    """
    if record_count < 500:
        raise ValueError(
            "REQ-011 violation: Dataset must contain at least 500 records."
        )

    rng = np.random.default_rng(random_seed)

    dataset = pd.DataFrame(
        {
            "complaint_id": range(1, record_count + 1),
            "product": rng.choice(ALLOWED_PRODUCTS, size=record_count),
            "category": rng.choice(ALLOWED_CATEGORIES, size=record_count),
            "initial_severity": rng.integers(1, 6, size=record_count),
            "days_open": rng.integers(1, 91, size=record_count),
            "repeat_event": rng.choice([0, 1], size=record_count),
            "vip_customer": rng.choice([0, 1], size=record_count),
        }
    )

    impact_scores = (
        dataset["initial_severity"] * 2
        + dataset["days_open"] / 20
        + dataset["repeat_event"] * 2
        + dataset["vip_customer"]
    )

    dataset["impact"] = np.select(
        [
            impact_scores < 5,
            (impact_scores >= 5) & (impact_scores < 9),
            impact_scores >= 9,
        ],
        [
            "Low",
            "Medium",
            "High",
        ],
        default="Medium",
    )

    validate_dataset(dataset)

    LOGGER.info(
        "Generated synthetic dataset with %s records using seed %s",
        len(dataset),
        random_seed,
    )

    return dataset


def validate_dataset(dataset: pd.DataFrame) -> None:
    """
    Validate dataset schema and content.

    REQ Coverage:
    - REQ-011
    - REQ-012

    Args:
        dataset: Dataset to validate.

    Raises:
        ValueError: If validation fails.
    """
    required_columns = {
        "complaint_id",
        "product",
        "category",
        "initial_severity",
        "days_open",
        "repeat_event",
        "vip_customer",
        "impact",
    }

    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        raise ValueError(
            f"REQ-012 violation: Missing columns {missing_columns}"
        )

    if len(dataset) < 500:
        raise ValueError(
            "REQ-011 violation: Dataset contains fewer than 500 records."
        )

    if dataset["complaint_id"].duplicated().any():
        raise ValueError(
            "REQ-012 violation: Duplicate complaint_id values detected."
        )

    invalid_impacts = set(dataset["impact"]) - set(ALLOWED_IMPACTS)

    if invalid_impacts:
        raise ValueError(
            f"REQ-012 violation: Invalid impact values {invalid_impacts}"
        )


def save_dataset(
    dataset: pd.DataFrame,
    output_file: str,
) -> None:
    """
    Save dataset to a CSV file.

    Args:
        dataset: Dataset to save.
        output_file: Output file path.
    """
    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    LOGGER.info(
        "Dataset saved to %s",
        output_path,
    )


def main() -> None:
    """
    Generate and save the default project dataset.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dataset = generate_dataset()

    save_dataset(
        dataset=dataset,
        output_file="data/raw/complaints_dataset_v1.csv",
    )


if __name__ == "__main__":
    main()
