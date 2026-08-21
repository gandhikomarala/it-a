# Multi-Cloud Infrastructure: Commercial BIM Construction Tracking
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "commercial_construction_bim_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Commercial BIM Construction Tracking"
}

resource "aws_kinesis_stream" "commercial_construction_bim_stream" {
  name             = "${var.environment}-commercial_construction_bim-stream"
  shard_count      = 2
  retention_period = var.commercial_construction_bim_retention_days * 24
  tags             = { Domain = "commercial_construction_bim" }
}
