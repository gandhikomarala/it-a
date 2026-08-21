# AWS Terraform Module: SNS
# Amazon SNS notification topics for critical drift alerts and operational emergencies.

resource "aws_sns_topic" "alerts" {
  name = "${var.environment}-churn-alerts-topic"
}
resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = "mlops-alerts@enterprise-mlops.io"
}
