# Multi-Cloud Infrastructure: Veterinary Practice Management
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "veterinary_practice_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Veterinary Practice Management"
}

resource "aws_kinesis_stream" "veterinary_practice_stream" {
  name             = "${var.environment}-veterinary_practice-stream"
  shard_count      = 2
  retention_period = var.veterinary_practice_retention_days * 24
  tags             = { Domain = "veterinary_practice" }
}
