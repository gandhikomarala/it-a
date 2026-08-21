# Multi-Cloud Infrastructure: Semiconductor 3nm Wafer Fab Yield
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "semiconductor_fab_yield_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Semiconductor 3nm Wafer Fab Yield"
}

resource "aws_kinesis_stream" "semiconductor_fab_yield_stream" {
  name             = "${var.environment}-semiconductor_fab_yield-stream"
  shard_count      = 2
  retention_period = var.semiconductor_fab_yield_retention_days * 24
  tags             = { Domain = "semiconductor_fab_yield" }
}
