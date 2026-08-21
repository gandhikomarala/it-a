# AWS Terraform Module: KMS
# Customer Managed Keys (CMK) with automated rotation for database and storage encryption.

resource "aws_kms_key" "churn_key" {
  description             = "KMS Key for Customer Churn MLOps platform encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags                    = { Name = "${var.environment}-churn-kms" }
}
resource "aws_kms_alias" "churn_key_alias" {
  name          = "alias/${var.environment}-churn-key"
  target_key_id = aws_kms_key.churn_key.key_id
}
