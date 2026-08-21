# Multi-Cloud Infrastructure: Wholesale Colocation Datacenter Power
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "colocation_datacenter_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Wholesale Colocation Datacenter Power"
}

resource "aws_kinesis_stream" "colocation_datacenter_stream" {
  name             = "${var.environment}-colocation_datacenter-stream"
  shard_count      = 2
  retention_period = var.colocation_datacenter_retention_days * 24
  tags             = { Domain = "colocation_datacenter" }
}
