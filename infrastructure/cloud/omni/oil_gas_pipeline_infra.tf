# Multi-Cloud Infrastructure: Oil & Gas Pipeline Integrity
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "oil_gas_pipeline_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Oil & Gas Pipeline Integrity"
}

resource "aws_kinesis_stream" "oil_gas_pipeline_stream" {
  name             = "${var.environment}-oil_gas_pipeline-stream"
  shard_count      = 2
  retention_period = var.oil_gas_pipeline_retention_days * 24
  tags             = { Domain = "oil_gas_pipeline" }
}
