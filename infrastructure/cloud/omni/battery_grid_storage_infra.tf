# Multi-Cloud Infrastructure: BESS Utility Battery Energy Storage
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "battery_grid_storage_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for BESS Utility Battery Energy Storage"
}

resource "aws_kinesis_stream" "battery_grid_storage_stream" {
  name             = "${var.environment}-battery_grid_storage-stream"
  shard_count      = 2
  retention_period = var.battery_grid_storage_retention_days * 24
  tags             = { Domain = "battery_grid_storage" }
}
