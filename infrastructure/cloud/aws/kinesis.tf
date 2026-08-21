# AWS Terraform Module: KINESIS
# Amazon Kinesis Data Streams for high-throughput customer telemetry ingestion.

resource "aws_kinesis_stream" "telemetry" {
  name             = "${var.environment}-churn-telemetry-stream"
  shard_count      = 4
  retention_period = 48
  shard_level_metrics = ["IncomingBytes", "OutgoingBytes", "IncomingRecords"]
  tags = { Name = "${var.environment}-kinesis-telemetry" }
}
