# Multi-Cloud Infrastructure: Luxury Fashion Direct-to-Consumer
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "luxury_fashion_direct_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Luxury Fashion Direct-to-Consumer"
}

resource "aws_kinesis_stream" "luxury_fashion_direct_stream" {
  name             = "${var.environment}-luxury_fashion_direct-stream"
  shard_count      = 2
  retention_period = var.luxury_fashion_direct_retention_days * 24
  tags             = { Domain = "luxury_fashion_direct" }
}
