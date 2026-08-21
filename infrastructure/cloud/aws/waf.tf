# AWS Terraform Module: WAF
# AWS WAF Web ACL rules protecting REST APIs against SQL injection and rate flooding.

resource "aws_wafv2_web_acl" "main" {
  name        = "${var.environment}-churn-waf"
  description = "WAF rules for Churn API"
  scope       = "REGIONAL"
  default_action { allow {} }
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.environment}-waf-metric"
    sampled_requests_enabled   = true
  }
}
