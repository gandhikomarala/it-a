# Multi-Cloud Infrastructure: Beverage CPG Direct Store Delivery
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "beverage_cpg_distrib_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Beverage CPG Direct Store Delivery"
}

resource "aws_kinesis_stream" "beverage_cpg_distrib_stream" {
  name             = "${var.environment}-beverage_cpg_distrib-stream"
  shard_count      = 2
  retention_period = var.beverage_cpg_distrib_retention_days * 24
  tags             = { Domain = "beverage_cpg_distrib" }
}
