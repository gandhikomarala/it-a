# Sliding window event aggregator for real-time feature computation.
from typing import List, Dict, Any
import time
import numpy as np

class SlidingWindowAggregator:
    @staticmethod
    def aggregate_events(events: List[Dict[str, Any]], window_seconds: float = 3600.0) -> Dict[str, Any]:
        now = time.time()
        recent_events = [e for e in events if (now - e.get("timestamp", now)) <= window_seconds]
        
        counts_by_type = {}
        for e in recent_events:
            t = e.get("event_type", "generic")
            counts_by_type[t] = counts_by_type.get(t, 0) + 1

        return {
            "window_seconds": window_seconds,
            "total_event_count": len(recent_events),
            "event_frequency_per_minute": len(recent_events) / (window_seconds / 60.0),
            "event_breakdown": counts_by_type
        }
