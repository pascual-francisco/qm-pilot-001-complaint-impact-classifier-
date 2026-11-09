"""
train_model.py

REQ Coverage:
- REQ-002: Train a classification model using the approved dataset.
- REQ-015: Execute a controlled training procedure.

Risk Coverage:
- RISK-005: Model training failure.

Verification:
- OQ-005
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_TYPE: Final[str] = "random_forest"
DEFAULT_RANDOM_SEED: Final[int] = 42

VALID_MODELS: Final[set[str]] = {
    "logistic_regression",
    "random_forest",
}


def load_training_dataset(
    input_file: str,
) -> pd.DataFrame:
    """
    Load the training dataset.

    Args:
        input_file: Path to training data.

    Returns:
        Training dataframe.
    """
    return pd.read_csv(input_file)


def prepare_training_data(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target.

    Args:
        dataset: Training dataset.

    Returns:
        Features and target.
    """
    training_data = dataset.copy()

    categorical_columns = [
        "product",
        "category",
    ]

    training_data = pd.get_dummies(
        training_data,
        columns=categorical_columns,
        drop_first=False,
    )

    target = training_data["impact"]

    features = training_data.drop(
        columns=[
            "impact",
            "complaint_id",
        ]
    )

    return features, target


def create_model(
    model_type: str,
):
    """
    Create approved model.

    Args:
        model_type: Model identifier.

    Returns:
        Instantiated model.
    """
    if model_type not in VALID_MODELS:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    if model_type == "logistic_regression":
        return LogisticRegression(
            random_state=DEFAULT_RANDOM_SEED,
            max_iter=1000,
        )

    return RandomForestClassifier(
        random_state=DEFAULT_RANDOM_SEED,
        n_estimators=100,
    )


def train_model(
    dataset: pd.DataFrame,
    model_type: str = DEFAULT_MODEL_TYPE,
):
    """
    Train the selected model.

    Args:
        dataset: Training dataset.
        model_type: Model identifier.

    Returns:
        Trained model.
    """
    features, target = prepare_training_data(dataset)

    model = create_model(model_type)

    model.fit(
        features,
        target,
    )

    LOGGER.info(
        "Training completed using %s",
        model_type,
    )

    return model


def save_model(
    model,
    model_type: str,
) -> None:
    """
    Save trained model and metadata.

    Args:
        model: Trained model.
        model_type: Model identifier.
    """
    model_dir = Path("models/v1.0.0")

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = model_dir / "model.pkl"

    joblib.dump(
        model,
        model_path,
    )

    metadata = {
        "model_type": model_type,
        "version": "1.0.0",
        "random_seed": DEFAULT_RANDOM_SEED,
    }

    with open(
        model_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    LOGGER.info(
        "Model saved to %s",
        model_path,
    )


def main() -> None:
    """
    Execute training workflow.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dataset = load_training_dataset(
        "data/train/complaints_train.csv"
    )

    model = train_model(
        dataset=dataset,
        model_type=DEFAULT_MODEL_TYPE,
    )

    save_model(
        model=model,
        model_type=DEFAULT_MODEL_TYPE,
    )


if __name__ == "__main__":
    main()
