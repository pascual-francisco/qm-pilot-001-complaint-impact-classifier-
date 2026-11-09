# RMP-QM-PILOT-001 Risk Management Plan - Complaint Impact Classifier

| Field | Value |
|---|---|
| Doc ID | RMP-QM-PILOT-001 |
| Title | Risk Management Plan - Complaint Impact Classifier |
| Version | 0.1.0 |
| Status | Draft - pending approval |
| Owner | Francisco Pascual |
| Approver | Francisco Pascual |
| Effective date | Not effective until approved by DIRECTOR |

> Created according to SOP-006, WI-010, and WI-033. This project uses synthetic data only.

## 1. Scope and intended use

This risk management plan covers the Complaint Impact Classifier pilot project. The project objective is to develop, validate, and release a machine learning model capable of classifying synthetic customer complaints into predefined impact categories (Low, Medium, High).

The primary purpose of the project is to validate the execution of the S.T.A.N.D.A.R.D. Quality System in a controlled environment using synthetic data only.

## 2. Risk categories considered

- Functional risks
- Data quality risks
- Security and privacy risks
- Machine learning risks
  - Hallucination
  - Bias
  - Data leakage
  - Prompt injection (where applicable)
  - Dataset drift
  - Feature leakage
  - Incorrect classification
- Validation risks
- Traceability risks
- Configuration management risks

## 3. Scales

Severity (S), Occurrence (O), and Detection (D) will use the default SOP-006 scales.

| Score | Severity | Occurrence | Detection |
|---|---|---|---|
| 1-3 | Minor inconvenience | Rare | Almost certainly detected |
| 4-6 | Wrong result a user might act on | Occasional | May be detected |
| 7-10 | Data leak, harmful or unrecoverable outcome | Frequent | Unlikely to be detected |

Occurrence scoring shall use available project evidence where applicable.

Scores of 7 or greater shall be justified and documented.

## 4. FMEA

The project FMEA shall be maintained in:

```text
docs/quality/FMEA-QM-PILOT-001.csv
```

The FMEA shall be completed according to WI-033.

All identified failure modes shall receive:

- Severity score
- Occurrence score
- Detection score
- Calculated RPN
- Defined control (if required)
- Linked requirement
- Verification test

## 5. Acceptability criteria

The project adopts the default SOP-006 thresholds:

- RPN ≥ 100 requires a documented control.
- Severity ≥ 8 requires a documented control regardless of RPN.

These thresholds are team conventions and are not claimed to originate from ISO 14971.

Residual risk may only be accepted by the DIRECTOR after review of verification evidence.

## 6. Controls

Every required control shall:

- Be converted into a controlled requirement (REQ).
- Receive a unique REQ identifier.
- Be linked to at least one verification test.
- Be included in project traceability records.

Verification of controls shall occur during OQ and/or PQ activities.

## 7. Review points

The Risk Management Plan and FMEA shall be reviewed:

- Before design freeze.
- Before release.
- After a deviation.
- After a CAPA.
- After an A3 investigation.
- After any significant project change.

## 8. FMEA file

```text
docs/quality/FMEA-QM-PILOT-001.csv
```

## Roles

| Role | Responsibility |
|---|---|
| DIRECTOR | Approves risk acceptance criteria and residual risks |
| Project Manager / Quality Coordinator | Coordinates risk activities, scoring, controls, and traceability |
| Software Engineer | Identifies failure modes, implements controls, and performs verification |
| Software Quality Engineer | Independently reviews risk records and supporting evidence |

## Revision History

| Version | Date | Author | Description of change | Approved by |
|---|---|---|---|---|
| 0.1.0 | 2026-11-08 | Software Engineer (AI-assisted) | Initial draft for QM-PILOT-001 Complaint Impact Classifier | Pending DIRECTOR Approval |
