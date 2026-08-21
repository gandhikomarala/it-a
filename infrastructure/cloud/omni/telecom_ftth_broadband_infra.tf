# Multi-Cloud Infrastructure: FTTH Fiber Gigabit Broadband Access
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "telecom_ftth_broadband_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for FTTH Fiber Gigabit Broadband Access"
}

resource "aws_kinesis_stream" "telecom_ftth_broadband_stream" {
  name             = "${var.environment}-telecom_ftth_broadband-stream"
  shard_count      = 2
  retention_period = var.telecom_ftth_broadband_retention_days * 24
  tags             = { Domain = "telecom_ftth_broadband" }
}
