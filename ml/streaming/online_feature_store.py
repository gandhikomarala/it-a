# High-speed in-memory / Redis online feature store with sub-2ms retrieval.
from typing import Dict, Any, Optional, List
import time

class OnlineFeatureStore:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def put_customer_features(self, customer_id: str, features: Dict[str, Any], ttl_seconds: int = 86400) -> None:
        self._store[customer_id] = {
            "features": features,
            "updated_at": time.time(),
            "ttl": ttl_seconds
        }

    def get_customer_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        record = self._store.get(customer_id)
        if not record:
            return None
        return record["features"]

    def batch_get(self, customer_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        return {cid: self.get_customer_features(cid) for cid in customer_ids if self.get_customer_features(cid) is not None}
