# Multi-Cloud Infrastructure: Patent Prosecution & IP Portfolio
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "legal_ip_patents_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Patent Prosecution & IP Portfolio"
}

resource "aws_kinesis_stream" "legal_ip_patents_stream" {
  name             = "${var.environment}-legal_ip_patents-stream"
  shard_count      = 2
  retention_period = var.legal_ip_patents_retention_days * 24
  tags             = { Domain = "legal_ip_patents" }
}
