# Multi-Cloud Infrastructure: Airline Yield & Revenue Management
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "airline_revenue_mgmt_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Airline Yield & Revenue Management"
}

resource "aws_kinesis_stream" "airline_revenue_mgmt_stream" {
  name             = "${var.environment}-airline_revenue_mgmt-stream"
  shard_count      = 2
  retention_period = var.airline_revenue_mgmt_retention_days * 24
  tags             = { Domain = "airline_revenue_mgmt" }
}
