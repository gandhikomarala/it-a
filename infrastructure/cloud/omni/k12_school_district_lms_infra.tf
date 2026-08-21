# Multi-Cloud Infrastructure: K-12 School District LMS Analytics
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "k12_school_district_lms_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for K-12 School District LMS Analytics"
}

resource "aws_kinesis_stream" "k12_school_district_lms_stream" {
  name             = "${var.environment}-k12_school_district_lms-stream"
  shard_count      = 2
  retention_period = var.k12_school_district_lms_retention_days * 24
  tags             = { Domain = "k12_school_district_lms" }
}
