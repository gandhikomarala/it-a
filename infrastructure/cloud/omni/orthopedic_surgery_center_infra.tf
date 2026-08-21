# Multi-Cloud Infrastructure: Ambulatory Surgical Center Operations
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "orthopedic_surgery_center_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Ambulatory Surgical Center Operations"
}

resource "aws_kinesis_stream" "orthopedic_surgery_center_stream" {
  name             = "${var.environment}-orthopedic_surgery_center-stream"
  shard_count      = 2
  retention_period = var.orthopedic_surgery_center_retention_days * 24
  tags             = { Domain = "orthopedic_surgery_center" }
}
