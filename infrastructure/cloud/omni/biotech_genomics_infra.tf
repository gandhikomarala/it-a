# Multi-Cloud Infrastructure: Biotech & Next-Gen Sequencing SaaS
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "biotech_genomics_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Biotech & Next-Gen Sequencing SaaS"
}

resource "aws_kinesis_stream" "biotech_genomics_stream" {
  name             = "${var.environment}-biotech_genomics-stream"
  shard_count      = 2
  retention_period = var.biotech_genomics_retention_days * 24
  tags             = { Domain = "biotech_genomics" }
}
