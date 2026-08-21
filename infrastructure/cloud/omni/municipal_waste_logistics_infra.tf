# Multi-Cloud Infrastructure: Municipal Smart Waste Routing
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "municipal_waste_logistics_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Municipal Smart Waste Routing"
}

resource "aws_kinesis_stream" "municipal_waste_logistics_stream" {
  name             = "${var.environment}-municipal_waste_logistics-stream"
  shard_count      = 2
  retention_period = var.municipal_waste_logistics_retention_days * 24
  tags             = { Domain = "municipal_waste_logistics" }
}
