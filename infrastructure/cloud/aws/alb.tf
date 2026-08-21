# AWS Terraform Module: ALB
# Application Load Balancer with HTTPS listeners and target groups.

resource "aws_lb" "api" {
  name               = "${var.environment}-churn-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ecs.id]
  subnets            = [aws_subnet.public_1.id, aws_subnet.public_2.id]
  tags               = { Name = "${var.environment}-alb" }
}
resource "aws_lb_target_group" "api" {
  name        = "${var.environment}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  health_check {
    path                = "/api/v1/system/health"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}
