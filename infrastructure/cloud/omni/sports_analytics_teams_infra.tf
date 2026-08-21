# Multi-Cloud Infrastructure: Professional Sports Team Athlete Tracking
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "sports_analytics_teams_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Professional Sports Team Athlete Tracking"
}

resource "aws_kinesis_stream" "sports_analytics_teams_stream" {
  name             = "${var.environment}-sports_analytics_teams-stream"
  shard_count      = 2
  retention_period = var.sports_analytics_teams_retention_days * 24
  tags             = { Domain = "sports_analytics_teams" }
}
