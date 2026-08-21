# Multi-Cloud Infrastructure: Executive Leadership Coaching SaaS
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "executive_coaching_saas_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Executive Leadership Coaching SaaS"
}

resource "aws_kinesis_stream" "executive_coaching_saas_stream" {
  name             = "${var.environment}-executive_coaching_saas-stream"
  shard_count      = 2
  retention_period = var.executive_coaching_saas_retention_days * 24
  tags             = { Domain = "executive_coaching_saas" }
}
