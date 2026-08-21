# GCP Terraform Module: MEMORYSTORE
# Google Cloud Memorystore for Redis instance for low-latency feature caching.

resource "google_redis_instance" "cache" {
  name           = "${var.environment}-churn-redis"
  tier           = "STANDARD_HA"
  memory_size_gb = 5
  region         = var.gcp_region
}
