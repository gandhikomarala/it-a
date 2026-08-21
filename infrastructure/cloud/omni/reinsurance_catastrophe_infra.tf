# Multi-Cloud Infrastructure: Catastrophe Reinsurance Modeling
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "reinsurance_catastrophe_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Catastrophe Reinsurance Modeling"
}

resource "aws_kinesis_stream" "reinsurance_catastrophe_stream" {
  name             = "${var.environment}-reinsurance_catastrophe-stream"
  shard_count      = 2
  retention_period = var.reinsurance_catastrophe_retention_days * 24
  tags             = { Domain = "reinsurance_catastrophe" }
}
