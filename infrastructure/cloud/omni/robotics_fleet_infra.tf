# Multi-Cloud Infrastructure: Autonomous Robotics & AMR Fleet Management
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "robotics_fleet_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Autonomous Robotics & AMR Fleet Management"
}

resource "aws_kinesis_stream" "robotics_fleet_stream" {
  name             = "${var.environment}-robotics_fleet-stream"
  shard_count      = 2
  retention_period = var.robotics_fleet_retention_days * 24
  tags             = { Domain = "robotics_fleet" }
}
