# Multi-Cloud Infrastructure: Global CDN Video Edge Caching
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "cdn_edge_streaming_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Global CDN Video Edge Caching"
}

resource "aws_kinesis_stream" "cdn_edge_streaming_stream" {
  name             = "${var.environment}-cdn_edge_streaming-stream"
  shard_count      = 2
  retention_period = var.cdn_edge_streaming_retention_days * 24
  tags             = { Domain = "cdn_edge_streaming" }
}
