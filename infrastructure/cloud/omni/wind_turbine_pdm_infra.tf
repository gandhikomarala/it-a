# Multi-Cloud Infrastructure: Offshore Wind Turbine Predictive Maintenance
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "wind_turbine_pdm_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Offshore Wind Turbine Predictive Maintenance"
}

resource "aws_kinesis_stream" "wind_turbine_pdm_stream" {
  name             = "${var.environment}-wind_turbine_pdm-stream"
  shard_count      = 2
  retention_period = var.wind_turbine_pdm_retention_days * 24
  tags             = { Domain = "wind_turbine_pdm" }
}
