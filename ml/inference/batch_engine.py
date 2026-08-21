# High-throughput streaming batch inference engine.
import os
import time
from typing import Callable, Optional
import pandas as pd
import numpy as np
from packages.logging.logger import get_logger
from .risk_categorizer import RiskCategorizer

logger = get_logger(__name__)

class BatchInferenceEngine:
    def __init__(self, pipeline: Any, model_wrapper: Any, batch_size: int = 5000):
        self.pipeline = pipeline
        self.model_wrapper = model_wrapper
        self.batch_size = batch_size

    def process_file(
        self,
        input_filepath: str,
        output_filepath: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        logger.info(f"Starting batch prediction for {input_filepath} -> {output_filepath}")
        
        if input_filepath.endswith(".parquet"):
            df = pd.read_parquet(input_filepath)
        else:
            df = pd.read_csv(input_filepath)

        total_rows = len(df)
        processed_rows = 0
        results_list = []

        for i in range(0, total_rows, self.batch_size):
            chunk = df.iloc[i:i + self.batch_size].copy()
            cids = chunk["customer_id"].tolist() if "customer_id" in chunk.columns else [f"CUS-{k}" for k in range(i, i + len(chunk))]

            X_trans = self.pipeline.transform(chunk)
            probs = self.model_wrapper.predict_proba(X_trans)
            preds = (probs >= 0.50).astype(int)

            for cid, prob, pred in zip(cids, probs, preds):
                risk = RiskCategorizer.categorize(float(prob))
                results_list.append({
                    "customer_id": cid,
                    "prediction": int(pred),
                    "churn_probability": round(float(prob), 4),
                    "risk_level": risk.value,
                    "confidence": round(float(abs(prob - 0.5) * 2), 3)
                })

            processed_rows += len(chunk)
            if progress_callback:
                progress_callback(processed_rows, total_rows)

        results_df = pd.DataFrame(results_list)
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        
        if output_filepath.endswith(".parquet"):
            results_df.to_parquet(output_filepath, index=False)
        else:
            results_df.to_csv(output_filepath, index=False)

        logger.info(f"Batch prediction completed: {total_rows} records written to {output_filepath}")
        return total_rows
