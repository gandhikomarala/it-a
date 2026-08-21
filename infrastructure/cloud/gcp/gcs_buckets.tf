# GCP Terraform Module: GCS_BUCKETS
# Google Cloud Storage buckets for training datasets, model artifacts, and quality reports.

resource "google_storage_bucket" "artifacts" {
  name     = "${var.environment}-churn-ml-artifacts"
  location = var.gcp_region
  versioning { enabled = true }
}
