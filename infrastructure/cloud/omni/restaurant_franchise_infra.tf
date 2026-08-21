# Multi-Cloud Infrastructure: QSR Franchise Store Operations
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "restaurant_franchise_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for QSR Franchise Store Operations"
}

resource "aws_kinesis_stream" "restaurant_franchise_stream" {
  name             = "${var.environment}-restaurant_franchise-stream"
  shard_count      = 2
  retention_period = var.restaurant_franchise_retention_days * 24
  tags             = { Domain = "restaurant_franchise" }
}
