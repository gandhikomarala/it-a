# Multi-Cloud Infrastructure: Defense & Aerospace Telemetry
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "defense_aerospace_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Defense & Aerospace Telemetry"
}

resource "aws_kinesis_stream" "defense_aerospace_stream" {
  name             = "${var.environment}-defense_aerospace-stream"
  shard_count      = 2
  retention_period = var.defense_aerospace_retention_days * 24
  tags             = { Domain = "defense_aerospace" }
}
