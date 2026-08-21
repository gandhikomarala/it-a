# Multi-Cloud Infrastructure: Behavioral & Mental Health Telehealth
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "behavioral_mental_health_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Behavioral & Mental Health Telehealth"
}

resource "aws_kinesis_stream" "behavioral_mental_health_stream" {
  name             = "${var.environment}-behavioral_mental_health-stream"
  shard_count      = 2
  retention_period = var.behavioral_mental_health_retention_days * 24
  tags             = { Domain = "behavioral_mental_health" }
}
