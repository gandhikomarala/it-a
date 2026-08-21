# Multi-Cloud Infrastructure: University Admissions & Enrollment
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "higher_ed_admissions_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for University Admissions & Enrollment"
}

resource "aws_kinesis_stream" "higher_ed_admissions_stream" {
  name             = "${var.environment}-higher_ed_admissions-stream"
  shard_count      = 2
  retention_period = var.higher_ed_admissions_retention_days * 24
  tags             = { Domain = "higher_ed_admissions" }
}
