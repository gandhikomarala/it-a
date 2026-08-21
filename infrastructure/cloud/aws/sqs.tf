# AWS Terraform Module: SQS
# Amazon SQS Dead Letter Queues and asynchronous worker task queues.

resource "aws_sqs_queue" "task_queue" {
  name                      = "${var.environment}-churn-task-queue"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 86400
  receive_wait_time_seconds = 10
}
resource "aws_sqs_queue" "dlq" {
  name = "${var.environment}-churn-task-dlq"
}
