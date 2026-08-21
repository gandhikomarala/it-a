# Multi-Cloud Infrastructure: Cold Chain Biopharma Logistics
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "cold_chain_pharma_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Cold Chain Biopharma Logistics"
}

resource "aws_kinesis_stream" "cold_chain_pharma_stream" {
  name             = "${var.environment}-cold_chain_pharma-stream"
  shard_count      = 2
  retention_period = var.cold_chain_pharma_retention_days * 24
  tags             = { Domain = "cold_chain_pharma" }
}
