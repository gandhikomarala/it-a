# Multi-Cloud Infrastructure: Utility Solar Asset Performance
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "cleantech_solar_asset_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Utility Solar Asset Performance"
}

resource "aws_kinesis_stream" "cleantech_solar_asset_stream" {
  name             = "${var.environment}-cleantech_solar_asset-stream"
  shard_count      = 2
  retention_period = var.cleantech_solar_asset_retention_days * 24
  tags             = { Domain = "cleantech_solar_asset" }
}
