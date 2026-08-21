"""Central system constants, defaults, and error specifications."""

APP_NAME = "Enterprise Churn & MLOps Platform"
API_VERSION = "v1"
API_V1_STR = "/api/v1"

# Risk classification thresholds (probability of churn)
DEFAULT_RISK_THRESHOLDS = {
    "LOW_CUTOFF": 0.30,      # < 0.30 -> Low Risk
    "HIGH_CUTOFF": 0.70,     # >= 0.70 -> High Risk
    "CRITICAL_CUTOFF": 0.85  # >= 0.85 -> Critical Risk
}

# Drift detection threshold metrics
DRIFT_THRESHOLDS = {
    "PSI_NORMAL": 0.10,      # PSI < 0.10: No significant shift
    "PSI_WARNING": 0.25,     # 0.10 <= PSI < 0.25: Moderate shift
    "PSI_CRITICAL": 0.25,    # PSI >= 0.25: Significant drift detected
    "KS_TEST_P_VALUE": 0.05, # p-value < 0.05 indicates distribution divergence
    "WASSERSTEIN_LIMIT": 0.35,
    "PREDICTION_SHIFT_LIMIT": 0.15
}

# Quality score thresholds
QUALITY_SCORE_THRESHOLDS = {
    "EXCELLENT": 90.0,
    "GOOD": 80.0,
    "ACCEPTABLE": 70.0,
    "POOR": 50.0
}

# Automated Retraining Safeguards
RETRAINING_POLICIES = {
    "MIN_TEST_ROC_AUC": 0.75,
    "MIN_TEST_PR_AUC": 0.65,
    "MAX_PERFORMANCE_DROP_PCT": 0.05, # Candidate must not be >5% worse than current production
    "REQUIRED_SUPERIORITY_PCT": 0.01, # For auto-promotion, candidate must exceed production by >=1% ROC-AUC
    "MIN_TRAINING_SAMPLES": 500
}

# Standard error codes
SYSTEM_ERROR_CODES = {
    "VALIDATION_ERROR": "ERR_VALIDATION_FAILED",
    "AUTHENTICATION_ERROR": "ERR_AUTHENTICATION_REQUIRED",
    "AUTHORIZATION_ERROR": "ERR_PERMISSION_DENIED",
    "NOT_FOUND": "ERR_RESOURCE_NOT_FOUND",
    "CONFLICT": "ERR_RESOURCE_CONFLICT",
    "DATASET_ERROR": "ERR_DATASET_PROCESSING_FAILED",
    "MODEL_ERROR": "ERR_MODEL_EXECUTION_FAILED",
    "PREDICTION_ERROR": "ERR_PREDICTION_FAILED",
    "TRAINING_ERROR": "ERR_MODEL_TRAINING_FAILED",
    "DRIFT_ERROR": "ERR_DRIFT_COMPUTATION_FAILED",
    "STORAGE_ERROR": "ERR_ARTIFACT_STORAGE_FAILED",
    "RATE_LIMIT_EXCEEDED": "ERR_RATE_LIMIT_EXCEEDED",
    "INTERNAL_SERVER_ERROR": "ERR_INTERNAL_SYSTEM_FAILURE"
}

ROLE_PERMISSIONS_MAP = {
    "SUPER_ADMIN": [
        "customer:read", "customer:create", "customer:update", "customer:delete", "customer:export",
        "dataset:read", "dataset:upload", "dataset:validate", "dataset:delete", "dataset:approve",
        "feature:read", "feature:extract", "feature:manage",
        "experiment:read", "experiment:create", "experiment:delete",
        "model:read", "model:train", "model:deploy", "model:promote", "model:rollback", "model:archive",
        "prediction:read", "prediction:create", "prediction:batch",
        "analytics:read", "monitoring:read", "drift:check", "retraining:trigger",
        "report:read", "report:generate", "notification:manage",
        "user:read", "user:manage", "role:manage", "audit:read", "system:configure"
    ],
    "ADMIN": [
        "customer:read", "customer:create", "customer:update", "customer:delete", "customer:export",
        "dataset:read", "dataset:upload", "dataset:validate", "dataset:approve",
        "feature:read", "feature:extract", "feature:manage",
        "experiment:read", "experiment:create",
        "model:read", "model:train", "model:deploy", "model:promote", "model:rollback", "model:archive",
        "prediction:read", "prediction:create", "prediction:batch",
        "analytics:read", "monitoring:read", "drift:check", "retraining:trigger",
        "report:read", "report:generate", "notification:manage",
        "user:read", "user:manage", "audit:read"
    ],
    "ML_ENGINEER": [
        "customer:read", "customer:export",
        "dataset:read", "dataset:upload", "dataset:validate", "dataset:approve",
        "feature:read", "feature:extract", "feature:manage",
        "experiment:read", "experiment:create", "experiment:delete",
        "model:read", "model:train", "model:deploy", "model:promote", "model:rollback", "model:archive",
        "prediction:read", "prediction:create", "prediction:batch",
        "analytics:read", "monitoring:read", "drift:check", "retraining:trigger",
        "report:read", "report:generate",
        "audit:read"
    ],
    "DATA_ENGINEER": [
        "customer:read", "customer:create", "customer:update", "customer:export",
        "dataset:read", "dataset:upload", "dataset:validate", "dataset:approve", "dataset:delete",
        "feature:read", "feature:extract", "feature:manage",
        "model:read", "prediction:read",
        "analytics:read", "monitoring:read", "drift:check",
        "report:read", "report:generate"
    ],
    "ANALYST": [
        "customer:read", "customer:export",
        "dataset:read",
        "feature:read",
        "experiment:read",
        "model:read",
        "prediction:read", "prediction:create", "prediction:batch",
        "analytics:read", "monitoring:read", "drift:check",
        "report:read", "report:generate"
    ],
    "MANAGER": [
        "customer:read", "customer:export",
        "dataset:read",
        "model:read", "model:deploy", "model:promote",
        "prediction:read", "prediction:create", "prediction:batch",
        "analytics:read", "monitoring:read",
        "report:read", "report:generate",
        "audit:read"
    ],
    "VIEWER": [
        "customer:read",
        "dataset:read",
        "model:read",
        "prediction:read",
        "analytics:read",
        "monitoring:read",
        "report:read"
    ]
}
