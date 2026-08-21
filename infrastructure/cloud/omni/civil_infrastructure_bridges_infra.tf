# Multi-Cloud Infrastructure: Civil Infrastructure & Bridge Health
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "civil_infrastructure_bridges_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Civil Infrastructure & Bridge Health"
}

resource "aws_kinesis_stream" "civil_infrastructure_bridges_stream" {
  name             = "${var.environment}-civil_infrastructure_bridges-stream"
  shard_count      = 2
  retention_period = var.civil_infrastructure_bridges_retention_days * 24
  tags             = { Domain = "civil_infrastructure_bridges" }
}
