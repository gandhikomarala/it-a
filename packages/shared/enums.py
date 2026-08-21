"""Domain enumeration definitions for the Churn & MLOps Platform."""
from enum import Enum, unique

@unique
class UserRole(str, Enum):
    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    ML_ENGINEER = "ML_ENGINEER"
    DATA_ENGINEER = "DATA_ENGINEER"
    ANALYST = "ANALYST"
    MANAGER = "MANAGER"
    VIEWER = "VIEWER"

@unique
class Permission(str, Enum):
    # Customer permissions
    CUSTOMER_READ = "customer:read"
    CUSTOMER_CREATE = "customer:create"
    CUSTOMER_UPDATE = "customer:update"
    CUSTOMER_DELETE = "customer:delete"
    CUSTOMER_EXPORT = "customer:export"

    # Dataset permissions
    DATASET_READ = "dataset:read"
    DATASET_UPLOAD = "dataset:upload"
    DATASET_VALIDATE = "dataset:validate"
    DATASET_DELETE = "dataset:delete"
    DATASET_APPROVE = "dataset:approve"

    # Feature permissions
    FEATURE_READ = "feature:read"
    FEATURE_EXTRACT = "feature:extract"
    FEATURE_MANAGE = "feature:manage"

    # Experiment & Model permissions
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_CREATE = "experiment:create"
    EXPERIMENT_DELETE = "experiment:delete"

    MODEL_READ = "model:read"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_PROMOTE = "model:promote"
    MODEL_ROLLBACK = "model:rollback"
    MODEL_ARCHIVE = "model:archive"

    # Prediction permissions
    PREDICTION_READ = "prediction:read"
    PREDICTION_CREATE = "prediction:create"
    PREDICTION_BATCH = "prediction:batch"

    # Analytics & Monitoring permissions
    ANALYTICS_READ = "analytics:read"
    MONITORING_READ = "monitoring:read"
    DRIFT_CHECK = "drift:check"
    RETRAINING_TRIGGER = "retraining:trigger"

    # Reports & Notifications
    REPORT_READ = "report:read"
    REPORT_GENERATE = "report:generate"
    NOTIFICATION_MANAGE = "notification:manage"

    # Administration & Security
    USER_READ = "user:read"
    USER_MANAGE = "user:manage"
    ROLE_MANAGE = "role:manage"
    AUDIT_READ = "audit:read"
    SYSTEM_CONFIGURE = "system:configure"

@unique
class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@unique
class ModelStage(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"
    REJECTED = "REJECTED"

@unique
class TaskStatus(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"

@unique
class BatchStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@unique
class DriftStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@unique
class QualityScoreTier(str, Enum):
    EXCELLENT = "EXCELLENT"  # >= 90%
    GOOD = "GOOD"            # 80-89%
    ACCEPTABLE = "ACCEPTABLE" # 70-79%
    POOR = "POOR"            # 50-69%
    CRITICAL = "CRITICAL"    # < 50%

@unique
class NotificationType(str, Enum):
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    TRAINING_FAILED = "TRAINING_FAILED"
    MODEL_DEPLOYED = "MODEL_DEPLOYED"
    MODEL_ROLLBACK = "MODEL_ROLLBACK"
    MODEL_DEGRADED = "MODEL_DEGRADED"
    DATA_DRIFT_DETECTED = "DATA_DRIFT_DETECTED"
    BATCH_COMPLETED = "BATCH_COMPLETED"
    DATASET_VALIDATION_FAILED = "DATASET_VALIDATION_FAILED"
    SECURITY_ALERT = "SECURITY_ALERT"
    RETRAINING_TRIGGERED = "RETRAINING_TRIGGERED"

@unique
class NotificationChannel(str, Enum):
    IN_APP = "IN_APP"
    EMAIL = "EMAIL"
    WEBHOOK = "WEBHOOK"
    SLACK = "SLACK"

@unique
class AuditAction(str, Enum):
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    LOGIN_FAILED = "LOGIN_FAILED"
    PASSWORD_RESET = "PASSWORD_RESET"
    CREATE_USER = "CREATE_USER"
    UPDATE_USER = "UPDATE_USER"
    DELETE_USER = "DELETE_USER"
    ASSIGN_ROLE = "ASSIGN_ROLE"
    UPLOAD_DATASET = "UPLOAD_DATASET"
    VALIDATE_DATASET = "VALIDATE_DATASET"
    APPROVE_DATASET = "APPROVE_DATASET"
    DELETE_DATASET = "DELETE_DATASET"
    CREATE_EXPERIMENT = "CREATE_EXPERIMENT"
    TRAIN_MODEL = "TRAIN_MODEL"
    DEPLOY_MODEL = "DEPLOY_MODEL"
    PROMOTE_MODEL = "PROMOTE_MODEL"
    ROLLBACK_MODEL = "ROLLBACK_MODEL"
    ARCHIVE_MODEL = "ARCHIVE_MODEL"
    CREATE_PREDICTION = "CREATE_PREDICTION"
    CREATE_BATCH_JOB = "CREATE_BATCH_JOB"
    RUN_DRIFT_CHECK = "RUN_DRIFT_CHECK"
    TRIGGER_RETRAINING = "TRIGGER_RETRAINING"
    CHANGE_SETTINGS = "CHANGE_SETTINGS"
    GENERATE_REPORT = "GENERATE_REPORT"

@unique
class ContractType(str, Enum):
    MONTH_TO_MONTH = "Month-to-Month"
    ONE_YEAR = "One-Year"
    TWO_YEAR = "Two-Year"
    CUSTOM_ENTERPRISE = "Enterprise-Custom"

@unique
class PaymentMethod(str, Enum):
    CREDIT_CARD = "Credit Card"
    BANK_TRANSFER = "Bank Transfer"
    ELECTRONIC_CHECK = "Electronic Check"
    MAILED_CHECK = "Mailed Check"
    PAYPAL = "PayPal"
    CRYPTO = "Cryptocurrency"

@unique
class SubscriptionType(str, Enum):
    BASIC = "Basic"
    STANDARD = "Standard"
    PREMIUM = "Premium"
    ENTERPRISE = "Enterprise"
    CUSTOM = "Custom"

@unique
class SupportTicketPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"

@unique
class RetrainingTriggerType(str, Enum):
    MANUAL = "MANUAL"
    DRIFT_ALERT = "DRIFT_ALERT"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    SCHEDULED = "SCHEDULED"
    NEW_DATASET_APPROVED = "NEW_DATASET_APPROVED"
