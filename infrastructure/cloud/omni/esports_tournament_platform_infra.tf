# Multi-Cloud Infrastructure: Esports Tournament & Streaming Platform
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "esports_tournament_platform_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Esports Tournament & Streaming Platform"
}

resource "aws_kinesis_stream" "esports_tournament_platform_stream" {
  name             = "${var.environment}-esports_tournament_platform-stream"
  shard_count      = 2
  retention_period = var.esports_tournament_platform_retention_days * 24
  tags             = { Domain = "esports_tournament_platform" }
}
