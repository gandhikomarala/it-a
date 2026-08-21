# Multi-Cloud Infrastructure: Music Streaming Artist Royalty Split
# Dedicated streaming telemetry ingress and DynamoDB/Bigtable persistence layer.

variable "music_streaming_royalties_retention_days" {
  type        = number
  default     = 90
  description = "Telemetry data lake retention period in days for Music Streaming Artist Royalty Split"
}

resource "aws_kinesis_stream" "music_streaming_royalties_stream" {
  name             = "${var.environment}-music_streaming_royalties-stream"
  shard_count      = 2
  retention_period = var.music_streaming_royalties_retention_days * 24
  tags             = { Domain = "music_streaming_royalties" }
}
