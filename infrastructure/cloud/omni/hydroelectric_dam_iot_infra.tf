# Multi-Cloud Infrastructure: Hydroelectric Dam Structural Health
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "hydroelectric_dam_iot_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Hydroelectric Dam Structural Health"
}

resource "aws_kinesis_stream" "hydroelectric_dam_iot_stream" {
  name             = "${var.environment}-hydroelectric_dam_iot-stream"
  shard_count      = 2
  retention_period = var.hydroelectric_dam_iot_retention_days * 24
  tags             = { Domain = "hydroelectric_dam_iot" }
}
