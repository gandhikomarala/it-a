# Multi-Cloud Infrastructure: Dental DSO Practice Optimization
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "dental_dso_analytics_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Dental DSO Practice Optimization"
}

resource "aws_kinesis_stream" "dental_dso_analytics_stream" {
  name             = "${var.environment}-dental_dso_analytics-stream"
  shard_count      = 2
  retention_period = var.dental_dso_analytics_retention_days * 24
  tags             = { Domain = "dental_dso_analytics" }
}
