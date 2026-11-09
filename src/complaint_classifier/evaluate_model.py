"""
evaluate_model.py

REQ Coverage:
- REQ-004: Generate model evaluation metrics.
- REQ-016: Independently verify model performance calculations.
- REQ-017: Meet defined performance acceptance criteria.

Risk Coverage:
- RISK-006: Incorrect performance metrics.
- RISK-007: Model performance below acceptance criteria.

Verification:
- OQ-006
- PQ-001
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

LOGGER = logging.getLogger(__name__)

MODEL_PATH: Final[str] = "models/v1.0.0/model.pkl"
METRICS_PATH: Final[str] = "models/v1.0.0/metrics.json"

MINIMUM_ACCEPTABLE_ACCURACY: Final[float] = 0.80


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained model.

    Args:
        model_path: Path to model artifact.

    Returns:
        Trained model.
    """
    return joblib.load(model_path)


def load_test_dataset(
    input_file: str,
) -> pd.DataFrame:
    """
    Load testing dataset.

    Args:
        input_file: CSV path.

    Returns:
        DataFrame.
    """
    return pd.read_csv(input_file)


def prepare_evaluation_data(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert dataset into features and target.

    Args:
        dataset: Evaluation dataset.

    Returns:
        Features and target.
    """
    evaluation_data = dataset.copy()

    evaluation_data = pd.get_dummies(
        evaluation_data,
        columns=[
            "product",
            "category",
        ],
        drop_first=False,
    )

    target = evaluation_data["impact"]

    features = evaluation_data.drop(
        columns=[
            "impact",
            "complaint_id",
        ]
    )

    return features, target


def align_feature_columns(
    features: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Align evaluation features with training features.

    Args:
        features: Evaluation feature set.
        model: Trained model.

    Returns:
        Aligned feature dataframe.
    """
    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)

        features = features.reindex(
            columns=expected_columns,
            fill_value=0,
        )

    return features


def evaluate_model(
    model,
    dataset: pd.DataFrame,
) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained model.
        dataset: Test dataset.

    Returns:
        Evaluation metrics dictionary.
    """
    features, target = prepare_evaluation_data(dataset)

    features = align_feature_columns(
        features=features,
        model=model,
    )

    predictions = model.predict(features)

    accuracy = accuracy_score(
        target,
        predictions,
    )

    matrix = confusion_matrix(
        target,
        predictions,
    )

    report = classification_report(
        target,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    meets_acceptance = (
        accuracy >= MINIMUM_ACCEPTABLE_ACCURACY
    )

    results = {
        "accuracy": round(float(accuracy), 4),
        "acceptance_threshold": MINIMUM_ACCEPTABLE_ACCURACY,
        "pass": meets_acceptance,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    LOGGER.info(
        "Accuracy: %.4f | Pass: %s",
        accuracy,
        meets_acceptance,
    )

    return results


def save_metrics(
    results: dict,
    output_file: str = METRICS_PATH,
) -> None:
    """
    Save evaluation results.

    Args:
        results: Metrics dictionary.
        output_file: Output file path.
    """
    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    LOGGER.info(
        "Metrics saved to %s",
        output_path,
    )


def generate_evaluation_summary(
    results: dict,
) -> str:
    """
    Create evaluation summary text.

    Args:
        results: Evaluation metrics.

    Returns:
        Human-readable summary.
    """
    status = "PASS" if results["pass"] else "FAIL"

    return (
        f"Model Evaluation Result: {status}\n"
        f"Accuracy: {results['accuracy']:.4f}\n"
        f"Acceptance Threshold: "
        f"{results['acceptance_threshold']:.2f}\n"
    )


def main() -> None:
    """
    Execute evaluation workflow.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    model = load_model()

    dataset = load_test_dataset(
        "data/test/complaints_test.csv"
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    save_metrics(results)

    print(
        generate_evaluation_summary(results)
    )


if __name__ == "__main__":
    main()
