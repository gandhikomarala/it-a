# GCP Terraform Module: PUBSUB
# Google Cloud Pub/Sub topics for real-time customer event streaming.

resource "google_pubsub_topic" "events" {
  name = "${var.environment}-churn-events-topic"
}
