# Multi-Cloud Infrastructure: Corporate M&A Virtual Data Room
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "corporate_m_and_a_due_diligence_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Corporate M&A Virtual Data Room"
}

resource "aws_kinesis_stream" "corporate_m_and_a_due_diligence_stream" {
  name             = "${var.environment}-corporate_m_and_a_due_diligence-stream"
  shard_count      = 2
  retention_period = var.corporate_m_and_a_due_diligence_retention_days * 24
  tags             = { Domain = "corporate_m_and_a_due_diligence" }
}
