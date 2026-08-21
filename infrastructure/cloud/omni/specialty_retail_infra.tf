# Multi-Cloud Infrastructure: Specialty Retail Omnichannel Inventory
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "specialty_retail_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Specialty Retail Omnichannel Inventory"
}

resource "aws_kinesis_stream" "specialty_retail_stream" {
  name             = "${var.environment}-specialty_retail-stream"
  shard_count      = 2
  retention_period = var.specialty_retail_retention_days * 24
  tags             = { Domain = "specialty_retail" }
}
