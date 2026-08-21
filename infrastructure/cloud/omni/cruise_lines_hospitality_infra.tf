# Multi-Cloud Infrastructure: Cruise Line Passenger Lifetime Value
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "cruise_lines_hospitality_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Cruise Line Passenger Lifetime Value"
}

resource "aws_kinesis_stream" "cruise_lines_hospitality_stream" {
  name             = "${var.environment}-cruise_lines_hospitality-stream"
  shard_count      = 2
  retention_period = var.cruise_lines_hospitality_retention_days * 24
  tags             = { Domain = "cruise_lines_hospitality" }
}
