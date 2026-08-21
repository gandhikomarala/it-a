# Immutable model artifact storage, serialization, and hash verification.
import os
import joblib
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from packages.utilities.file_validator import calculate_file_sha256
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class ModelRegistryManager:
    def __init__(self, storage_dir: str = "./artifacts/models"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_model_artifact(
        self,
        model_name: str,
        version: int,
        pipeline: Any,
        model_wrapper: Any,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        filename = f"{model_name.lower()}_v{version}_{int(datetime.now(timezone.utc).timestamp())}.joblib"
        filepath = os.path.join(self.storage_dir, filename)

        bundle = {
            "model_name": model_name,
            "version": version,
            "pipeline": pipeline,
            "model": model_wrapper,
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        joblib.dump(bundle, filepath, compress=3)
        file_size = os.path.getsize(filepath)
        sha256 = calculate_file_sha256(filepath)

        logger.info(
            "Model artifact serialized",
            filepath=filepath,
            size_kb=round(file_size / 1024, 2),
            sha256=sha256[:12]
        )

        return {
            "file_path": filepath,
            "file_size_bytes": file_size,
            "checksum_sha256": sha256
        }

    def load_model_artifact(self, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found: {filepath}")
        return joblib.load(filepath)
