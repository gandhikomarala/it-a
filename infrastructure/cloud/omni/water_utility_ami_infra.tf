# Multi-Cloud Infrastructure: Municipal Smart Water AMI Network
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "water_utility_ami_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Municipal Smart Water AMI Network"
}

resource "aws_kinesis_stream" "water_utility_ami_stream" {
  name             = "${var.environment}-water_utility_ami-stream"
  shard_count      = 2
  retention_period = var.water_utility_ami_retention_days * 24
  tags             = { Domain = "water_utility_ami" }
}
