# Feature store abstraction for offline batch and online inference lookups.
from typing import Dict, List, Optional, Any
import pandas as pd
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class FeatureStore:
    def __init__(self):
        self._online_features: Dict[str, Dict[str, Any]] = {}

    def ingest_online_features(self, customer_id: str, features: Dict[str, Any]) -> None:
        self._online_features[customer_id] = features

    def get_online_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        return self._online_features.get(customer_id)

    def get_batch_features(self, customer_ids: List[str]) -> pd.DataFrame:
        records = []
        for cid in customer_ids:
            feat = self._online_features.get(cid, {})
            feat["customer_id"] = cid
            records.append(feat)
        return pd.DataFrame(records)
