# AWS Terraform Module: SECRETS
# AWS Secrets Manager credentials rotation for database and third-party APIs.

resource "aws_secretsmanager_secret" "api_secrets" {
  name                    = "${var.environment}-churn-api-secrets"
  recovery_window_in_days = 0
}
