# Streaming CSV and Parquet export engine.
import io
import pandas as pd
from typing import List, Dict, Any

class ExportService:
    @staticmethod
    def export_customers_to_csv(customers: List[Dict[str, Any]]) -> str:
        df = pd.DataFrame(customers)
        return df.to_csv(index=False)

    @staticmethod
    def export_predictions_to_csv(predictions: List[Dict[str, Any]]) -> str:
        df = pd.DataFrame(predictions)
        return df.to_csv(index=False)
