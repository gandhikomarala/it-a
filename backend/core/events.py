# Asynchronous Event Bus with pub/sub architecture.
import asyncio
from typing import Callable, Dict, List, Any
from packages.logging.logger import get_logger

logger = get_logger("event.bus")

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_name: str, handler: Callable):
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    async def publish(self, event_name: str, payload: Dict[str, Any]):
        handlers = self._subscribers.get(event_name, [])
        logger.info(f"Publishing event '{event_name}' to {len(handlers)} subscriber(s)", event=event_name)
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(payload))
                else:
                    handler(payload)
            except Exception as e:
                logger.error(f"Error executing event handler for '{event_name}': {e}", exc_info=True)

# Global event bus instance
event_bus = EventBus()
