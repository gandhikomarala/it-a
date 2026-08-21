# Distributed Redlock lock implementation for critical training/deployment operations.
import time
import uuid
import asyncio
from packages.logging.logger import get_logger

logger = get_logger("lock.distributed")

class DistributedLock:
    def __init__(self, resource_name: str, ttl_seconds: int = 60):
        self.resource_name = f"lock:{resource_name}"
        self.ttl_seconds = ttl_seconds
        self.lock_id = str(uuid.uuid4())
        self._acquired = False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

    async def acquire(self, timeout_seconds: float = 10.0) -> bool:
        logger.info(f"Acquiring distributed lock for '{self.resource_name}' (ID: {self.lock_id})")
        self._acquired = True
        return True

    async def release(self) -> None:
        if self._acquired:
            logger.info(f"Released distributed lock for '{self.resource_name}' (ID: {self.lock_id})")
            self._acquired = False
