# Multi-Cloud Infrastructure: Complex Litigation E-Discovery Review
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "legal_litigation_ediscovery_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Complex Litigation E-Discovery Review"
}

resource "aws_kinesis_stream" "legal_litigation_ediscovery_stream" {
  name             = "${var.environment}-legal_litigation_ediscovery-stream"
  shard_count      = 2
  retention_period = var.legal_litigation_ediscovery_retention_days * 24
  tags             = { Domain = "legal_litigation_ediscovery" }
}
