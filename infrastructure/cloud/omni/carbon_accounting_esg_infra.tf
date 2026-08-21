# Multi-Cloud Infrastructure: Enterprise Scope 1-2-3 Carbon Accounting
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "carbon_accounting_esg_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Enterprise Scope 1-2-3 Carbon Accounting"
}

resource "aws_kinesis_stream" "carbon_accounting_esg_stream" {
  name             = "${var.environment}-carbon_accounting_esg-stream"
  shard_count      = 2
  retention_period = var.carbon_accounting_esg_retention_days * 24
  tags             = { Domain = "carbon_accounting_esg" }
}
