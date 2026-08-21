# Multi-Cloud Infrastructure: Submarine Fiber Cable Capacity
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "submarine_cable_telecom_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Submarine Fiber Cable Capacity"
}

resource "aws_kinesis_stream" "submarine_cable_telecom_stream" {
  name             = "${var.environment}-submarine_cable_telecom-stream"
  shard_count      = 2
  retention_period = var.submarine_cable_telecom_retention_days * 24
  tags             = { Domain = "submarine_cable_telecom" }
}
