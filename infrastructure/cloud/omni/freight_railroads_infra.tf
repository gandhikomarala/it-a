# Multi-Cloud Infrastructure: Class I Freight Railroad Logistics
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "freight_railroads_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Class I Freight Railroad Logistics"
}

resource "aws_kinesis_stream" "freight_railroads_stream" {
  name             = "${var.environment}-freight_railroads-stream"
  shard_count      = 2
  retention_period = var.freight_railroads_retention_days * 24
  tags             = { Domain = "freight_railroads" }
}
