# MLOps Operational Runbook

## 1. Data Ingestion & Quality Gates
- All incoming customer cohorts are evaluated across 4 pillars: Completeness, Validity, Uniqueness, Consistency.
- Minimum quality threshold for automated training is 75.0%.

## 2. Model Training & Promotion
- Models are trained using 5-fold Stratified Cross-Validation.
- Promotion to Production requires Candidate ROC-AUC >= 0.75 and superiority >= 1.0% over active production.

## 3. Drift Monitoring & Alerting
- Population Stability Index (PSI) is calculated hourly:
  - PSI < 0.10: Normal
  - 0.10 <= PSI < 0.25: Moderate Shift (Warning)
  - PSI >= 0.25: Significant Drift (Triggers automated retraining)
