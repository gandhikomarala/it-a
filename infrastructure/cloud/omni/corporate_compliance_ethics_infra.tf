# Multi-Cloud Infrastructure: Enterprise Ethics Hotline & Whistleblower
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "corporate_compliance_ethics_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Enterprise Ethics Hotline & Whistleblower"
}

resource "aws_kinesis_stream" "corporate_compliance_ethics_stream" {
  name             = "${var.environment}-corporate_compliance_ethics-stream"
  shard_count      = 2
  retention_period = var.corporate_compliance_ethics_retention_days * 24
  tags             = { Domain = "corporate_compliance_ethics" }
}
