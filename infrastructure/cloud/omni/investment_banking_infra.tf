# Multi-Cloud Infrastructure: Investment Banking M&A Deal Pipeline
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "investment_banking_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Investment Banking M&A Deal Pipeline"
}

resource "aws_kinesis_stream" "investment_banking_stream" {
  name             = "${var.environment}-investment_banking-stream"
  shard_count      = 2
  retention_period = var.investment_banking_retention_days * 24
  tags             = { Domain = "investment_banking" }
}
