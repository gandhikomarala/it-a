# Multi-Cloud Infrastructure: Commercial Treasury & Syndicated Lending
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "commercial_banking_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Commercial Treasury & Syndicated Lending"
}

resource "aws_kinesis_stream" "commercial_banking_stream" {
  name             = "${var.environment}-commercial_banking-stream"
  shard_count      = 2
  retention_period = var.commercial_banking_retention_days * 24
  tags             = { Domain = "commercial_banking" }
}
