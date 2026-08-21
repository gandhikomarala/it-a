# Multi-Cloud Infrastructure: Surface Mount PCB Assembly Quality
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "pcb_smt_assembly_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Surface Mount PCB Assembly Quality"
}

resource "aws_kinesis_stream" "pcb_smt_assembly_stream" {
  name             = "${var.environment}-pcb_smt_assembly-stream"
  shard_count      = 2
  retention_period = var.pcb_smt_assembly_retention_days * 24
  tags             = { Domain = "pcb_smt_assembly" }
}
