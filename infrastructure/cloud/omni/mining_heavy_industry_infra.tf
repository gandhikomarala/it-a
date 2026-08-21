# Multi-Cloud Infrastructure: Mining & Heavy Equipment IoT
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "mining_heavy_industry_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Mining & Heavy Equipment IoT"
}

resource "aws_kinesis_stream" "mining_heavy_industry_stream" {
  name             = "${var.environment}-mining_heavy_industry-stream"
  shard_count      = 2
  retention_period = var.mining_heavy_industry_retention_days * 24
  tags             = { Domain = "mining_heavy_industry" }
}
