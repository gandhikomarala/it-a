# Multi-Cloud Infrastructure: Optometry & Optical Retail Chain
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "optometry_vision_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Optometry & Optical Retail Chain"
}

resource "aws_kinesis_stream" "optometry_vision_stream" {
  name             = "${var.environment}-optometry_vision-stream"
  shard_count      = 2
  retention_period = var.optometry_vision_retention_days * 24
  tags             = { Domain = "optometry_vision" }
}
