# Multi-Cloud Infrastructure: Private Wealth Advisory & Estate Planning
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "wealth_advisory_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Private Wealth Advisory & Estate Planning"
}

resource "aws_kinesis_stream" "wealth_advisory_stream" {
  name             = "${var.environment}-wealth_advisory-stream"
  shard_count      = 2
  retention_period = var.wealth_advisory_retention_days * 24
  tags             = { Domain = "wealth_advisory" }
}
