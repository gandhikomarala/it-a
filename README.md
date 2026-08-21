# Enterprise-Grade Customer Churn Prediction & MLOps Platform

A production-style, modular, high-scale Customer Churn Prediction and MLOps Platform architected with Python, FastAPI, React, TypeScript, Scikit-learn, LightGBM, SHAP, PostgreSQL, Redis, Celery, Docker, and AWS.

## Core Capabilities
- **Automated Data Ingestion & Validation**: Multi-format parsing (CSV, Parquet, JSON), schema inspection, outlier detection, data quality scoring.
- **Advanced Feature Engineering**: Domain-driven behavioral features (tenure dynamics, usage decay, support velocity, payment risk score, engagement index).
- **Multi-Model Machine Learning Engine**: Logistic Regression, Random Forest, Gradient Boosting, LightGBM, Calibrated Ensembles.
- **Explainable AI (XAI)**: SHAP-powered local factor attribution and global feature importance.
- **Enterprise Model Registry & Deployment**: Immutable model versioning, artifact hashing, stage promotion, and zero-downtime rollback.
- **Real-time & High-Throughput Batch Inference**: Sub-20ms single prediction API, streaming chunked batch processing via Celery.
- **Statistical Data & Prediction Drift Monitoring**: Continuous PSI, KS-test, Chi-square test, and automated drift alerting.
- **Automated Retraining Engine**: Configurable trigger policies with production comparison safeguards.
- **Enterprise RBAC & Security**: 7-tier role-based access control, granular permission checks, Argon2 password hashing, JWT token rotation.
- **Modern Observability**: Prometheus metrics exporter, Grafana dashboards, structured JSON logging.
- **Developer & Ops Tooling**: `churnctl` CLI, Docker Compose orchestration, AWS Terraform/CloudFormation templates, and exhaustive tests.
