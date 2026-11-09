
# URS-QM-PILOT-001 User Requirements Specification - Complaint Impact Classifier

| Field | Value |
|---|---|
| Doc ID | URS-QM-PILOT-001 |
| Title | User Requirements Specification - Complaint Impact Classifier |
| Version | 0.1.0 |
| Status | Draft - pending approval |
| Owner | Francisco Pascual |
| Approver | Francisco Pascual |
| Effective date | Not effective until approved by DIRECTOR |

> Created according to WI-001 and SOP-001. This project uses synthetic data only.

## Problem / intended use

The purpose of this project is to develop a machine learning system capable of classifying synthetic customer complaints into predefined impact categories (Low, Medium, High).

The intended users are portfolio developers, reviewers, and quality-system participants who need a controlled project to validate the execution of the S.T.A.N.D.A.R.D. Quality System. The primary objective is to validate process execution, traceability, risk control, governance, and evidence generation rather than maximize predictive performance.

| ID | Requirement (The user shall ...) | Acceptance criterion | Priority | Source |
|---|---|---|---|---|
| URS-001 | The user shall be able to generate a synthetic complaint dataset for model development and testing. | A dataset containing at least 500 synthetic complaint records can be generated successfully. | Must | Project objective |
| URS-002 | The user shall be able to train a machine learning classification model using the approved synthetic dataset. | The training process completes successfully without execution errors. | Must | Intended use |
| URS-003 | The user shall be able to classify complaints into Low, Medium, or High impact categories. | The model produces one and only one valid impact category for every evaluated record. | Must | Intended use |
| URS-004 | The user shall be able to evaluate model performance on an independent test dataset. | Accuracy, confusion matrix, and classification metrics are generated for the test set. | Must | Validation need |
| URS-005 | The user shall be able to maintain traceability between user requirements, derived requirements, tests, and results. | Every approved URS item has at least one linked requirement and one verification activity in the traceability matrix. | Must | SOP-001 |
| URS-006 | The user shall be able to reproduce dataset generation and model evaluation results. | Dataset version, generation method, and random seed are documented and available. | Must | Reproducibility requirement |
| URS-007 | The user shall be able to perform project validation activities according to the defined quality system. | IQ, OQ, and PQ records are created and completed according to the validation plan. | Must | SOP-003 |
| URS-008 | The user shall be able to review project evidence before release. | Validation records, evaluation reports, and traceability records are available for review before release approval. | Must | Release readiness |
| URS-009 | The user shall be able to manage project versions using controlled document and release practices. | Project artifacts are versioned and documented in the CHANGELOG before release. | Should | SOP-005 |
| URS-010 | The user shall be able to compare at least two candidate machine learning algorithms. | Evaluation results are available for a minimum of two algorithm candidates before final algorithm selection. | Should | Model selection need |

**Constraints:** synthetic data only; no employer data; no confidential information; no production deployment; no clinical, regulatory, or business decision making; all project artifacts shall remain under version control.

## Revision History

| Version | Date | Author | Description of change | Approved by |
|---|---|---|---|---|
| 0.1.0 | 2026-11-08 | Software Engineer (AI-assisted) | Initial draft for QM-PILOT-001 Complaint Impact Classifier | Pending DIRECTOR Approval |
