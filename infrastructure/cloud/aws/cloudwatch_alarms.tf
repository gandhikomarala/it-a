# AWS Terraform Module: CLOUDWATCH_ALARMS
# CloudWatch Metric Alarms for ECS CPU, memory, 5xx errors, and model latency.

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "${var.environment}-api-high-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
