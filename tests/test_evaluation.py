"""
test_evaluation.py

REQ Coverage:
- REQ-004
- REQ-016
- REQ-017

Risk Coverage:
- RISK-006
- RISK-007

Verification:
- OQ-006
- PQ-001
"""

from complaint_classifier.generate_dataset import generate_dataset
from complaint_classifier.train_model import train_model
from complaint_classifier.evaluate_model import (
    MINIMUM_ACCEPTABLE_ACCURACY,
    evaluate_model,
    generate_evaluation_summary,
    prepare_evaluation_data,
)


def test_req_004_evaluation_generates_required_metrics() -> None:
    """
    REQ-004 / OQ-006

    Verify evaluation metrics are generated.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    assert "accuracy" in results
    assert "confusion_matrix" in results
    assert "classification_report" in results


def test_req_016_accuracy_is_numeric() -> None:
    """
    REQ-016 / OQ-006

    Verify calculated accuracy is numeric.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    assert isinstance(
        results["accuracy"],
        float,
    )


def test_req_016_accuracy_within_valid_range() -> None:
    """
    REQ-016 / OQ-006

    Verify accuracy is between 0 and 1.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    assert 0.0 <= results["accuracy"] <= 1.0


def test_req_017_acceptance_flag_exists() -> None:
    """
    REQ-017 / PQ-001

    Verify pass/fail decision is generated.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    assert "pass" in results


def test_req_017_acceptance_threshold_matches_requirement() -> None:
    """
    REQ-017 / PQ-001

    Verify acceptance threshold is recorded.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    assert (
        results["acceptance_threshold"]
        == MINIMUM_ACCEPTABLE_ACCURACY
    )


def test_confusion_matrix_is_not_empty() -> None:
    """
    REQ-004 support test.

    Verify confusion matrix contains data.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    assert len(
        results["confusion_matrix"]
    ) > 0


def test_classification_report_contains_class_metrics() -> None:
    """
    REQ-004 support test.

    Verify classification report structure.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    report = results[
        "classification_report"
    ]

    assert "Low" in report
    assert "Medium" in report
    assert "High" in report


def test_prepare_evaluation_data_removes_target_column() -> None:
    """
    REQ-016 support test.

    Verify target column is not used as a feature.
    """
    dataset = generate_dataset()

    features, _ = prepare_evaluation_data(
        dataset
    )

    assert "impact" not in features.columns


def test_evaluation_summary_contains_result_text() -> None:
    """
    REQ-017 support test.

    Verify evaluation summary generation.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    summary = generate_evaluation_summary(
        results
    )

    assert "Accuracy" in summary


def test_evaluation_produces_boolean_pass_flag() -> None:
    """
    REQ-017 support test.

    Verify acceptance decision type.
    """
    dataset = generate_dataset()

    model = train_model(
        dataset=dataset,
        model_type="random_forest",
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
    )

    assert isinstance(
        results["pass"],
        bool,
    )
