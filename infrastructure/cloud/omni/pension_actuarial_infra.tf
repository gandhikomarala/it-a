# Multi-Cloud Infrastructure: Defined Benefit Pension Fund Actuarial
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "pension_actuarial_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Defined Benefit Pension Fund Actuarial"
}

resource "aws_kinesis_stream" "pension_actuarial_stream" {
  name             = "${var.environment}-pension_actuarial-stream"
  shard_count      = 2
  retention_period = var.pension_actuarial_retention_days * 24
  tags             = { Domain = "pension_actuarial" }
}
