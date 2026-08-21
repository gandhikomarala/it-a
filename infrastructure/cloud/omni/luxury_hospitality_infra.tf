# Multi-Cloud Infrastructure: Luxury Resort Concierge Guest Experience
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "luxury_hospitality_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Luxury Resort Concierge Guest Experience"
}

resource "aws_kinesis_stream" "luxury_hospitality_stream" {
  name             = "${var.environment}-luxury_hospitality-stream"
  shard_count      = 2
  retention_period = var.luxury_hospitality_retention_days * 24
  tags             = { Domain = "luxury_hospitality" }
}
