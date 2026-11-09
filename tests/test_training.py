"""
test_training.py

REQ Coverage:
- REQ-002
- REQ-015

Risk Coverage:
- RISK-005

Verification:
- OQ-005
"""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from complaint_classifier.generate_dataset import generate_dataset
from complaint_classifier.train_model import (
    VALID_MODELS,
    create_model,
    prepare_training_data,
    train_model,
)


def test_req_002_training_completes_successfully() -> None:
    """
    REQ-002 / OQ-005

    Verify the model training process completes.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    assert model is not None


def test_req_015_trained_model_has_fit_attributes() -> None:
    """
    REQ-015 / OQ-005

    Verify a trained model contains
    fitted model attributes.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    assert hasattr(
        model,
        "feature_importances_",
    )


def test_prepare_training_data_creates_features_and_target() -> None:
    """
    REQ-002 support test.

    Verify feature and target extraction.
    """
    dataset = generate_dataset()

    features, target = prepare_training_data(
        dataset
    )

    assert len(features) == len(target)

    assert "impact" not in features.columns


def test_create_random_forest_model() -> None:
    """
    REQ-015 support test.

    Verify approved Random Forest creation.
    """
    model = create_model(
        "random_forest"
    )

    assert isinstance(
        model,
        RandomForestClassifier,
    )


def test_create_logistic_regression_model() -> None:
    """
    REQ-015 support test.

    Verify approved Logistic Regression creation.
    """
    model = create_model(
        "logistic_regression"
    )

    assert isinstance(
        model,
        LogisticRegression,
    )


def test_invalid_model_type_is_rejected() -> None:
    """
    REQ-015 / OQ-005

    Verify unsupported models are rejected.
    """
    with pytest.raises(ValueError):
        create_model(
            "unsupported_model"
        )


def test_all_approved_model_types_are_registered() -> None:
    """
    REQ-015 support test.

    Verify approved model registry.
    """
    assert VALID_MODELS == {
        "logistic_regression",
        "random_forest",
    }


def test_training_works_for_logistic_regression() -> None:
    """
    REQ-002 / OQ-005

    Verify approved alternative model trains.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="logistic_regression",
    )

    assert model is not None


def test_training_dataset_not_empty_after_preparation() -> None:
    """
    REQ-002 support test.

    Verify feature matrix contains rows.
    """
    dataset = generate_dataset()

    features, _ = prepare_training_data(
        dataset
    )

    assert len(features) > 0


def test_training_target_contains_expected_classes() -> None:
    """
    REQ-002 support test.

    Verify training target classes exist.
    """
    dataset = generate_dataset()

    _, target = prepare_training_data(
        dataset
    )

    assert set(target.unique()) == {
        "Low",
        "Medium",
        "High",
    }
``
