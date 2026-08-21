# Unit Test for AlertNotificationStep.
import pytest
from ml.pipelines.steps.alert_notification_step import AlertNotificationStep

def test_alert_notification_step_execution():
    step = AlertNotificationStep()
    ctx = {"record_count": 50}
    res = step.execute(ctx)
    assert res["status"] == "COMPLETED"
    assert res["step"] == "alert_notification_step"
    assert res["metrics"]["records_processed"] == 50
