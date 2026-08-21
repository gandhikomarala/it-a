# Multi-Cloud Infrastructure: MVNO Mobile Virtual Network Operator
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "mvno_cellular_telecom_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for MVNO Mobile Virtual Network Operator"
}

resource "aws_kinesis_stream" "mvno_cellular_telecom_stream" {
  name             = "${var.environment}-mvno_cellular_telecom-stream"
  shard_count      = 2
  retention_period = var.mvno_cellular_telecom_retention_days * 24
  tags             = { Domain = "mvno_cellular_telecom" }
}
