# GCP Terraform Module: CLOUD_SQL
# Cloud SQL PostgreSQL 16 instance with automated daily backups and high availability.

resource "google_sql_database_instance" "main" {
  name             = "${var.environment}-churn-postgres"
  database_version = "POSTGRES_16"
  region           = var.gcp_region
  settings {
    tier = "db-custom-4-16384"
    availability_type = "REGIONAL"
    backup_configuration {
      enabled = true
      point_in_time_recovery_enabled = true
    }
  }
}
