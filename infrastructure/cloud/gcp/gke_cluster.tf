# GCP Terraform Module: GKE_CLUSTER
# Google Kubernetes Engine (GKE) Autopilot cluster for containerized microservices.

resource "google_container_cluster" "primary" {
  name     = "${var.environment}-churn-gke"
  location = var.gcp_region
  enable_autopilot = true
  ip_allocation_policy {}
}
