# Webhook dispatch service with HMAC signature and retry handling.
import hmac
import hashlib
import time
import json
from typing import Dict, Any
import httpx
from packages.logging.logger import get_logger

logger = get_logger("service.webhook")

class WebhookService:
    @staticmethod
    def generate_signature(secret: str, payload_bytes: bytes, timestamp: int) -> str:
        msg = f"{timestamp}.".encode("utf-8") + payload_bytes
        sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        return f"t={timestamp},v1={sig}"

    @staticmethod
    async def dispatch(url: str, secret: str, event_type: str, payload: Dict[str, Any]) -> bool:
        body = json.dumps({"event": event_type, "data": payload, "dispatched_at": int(time.time())}).encode("utf-8")
        timestamp = int(time.time())
        signature = WebhookService.generate_signature(secret, body, timestamp)

        headers = {
            "Content-Type": "application/json",
            "X-Churn-Signature": signature,
            "X-Churn-Event": event_type,
            "User-Agent": "Enterprise-Churn-Webhook-Dispatcher/1.0"
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, content=body, headers=headers)
                logger.info(f"Dispatched webhook '{event_type}' to {url}: Status {resp.status_code}")
                return resp.is_success
        except Exception as e:
            logger.error(f"Failed to dispatch webhook to {url}: {e}")
            return False
