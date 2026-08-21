# High-performance multi-format dataset loaders with memory optimization.
import os
import json
from typing import Generator, List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from packages.logging.logger import get_logger, LogContext

logger = get_logger(__name__)

class DataLoader:
    # Universal dataset loader supporting CSV, Parquet, and JSON formats.

    @staticmethod
    def load_dataframe(filepath: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        # Load dataset into Pandas DataFrame with type inference and downcasting.
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        with LogContext(logger, f"Load DataFrame from {ext} file", path=filepath):
            if ext == ".csv":
                df = pd.read_csv(filepath, nrows=max_rows)
            elif ext == ".parquet":
                if max_rows:
                    table = pq.read_table(filepath)
                    df = table.slice(0, max_rows).to_pandas()
                else:
                    df = pd.read_parquet(filepath)
            elif ext == ".json":
                df = pd.read_json(filepath, lines=True if filepath.endswith(".jsonl") else False)
                if max_rows:
                    df = df.head(max_rows)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Optimize memory usage
            df = DataLoader.optimize_dtypes(df)
            logger.info("Dataset loaded successfully", rows=len(df), cols=len(df.columns))
            return df

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        # Downcast numeric columns to reduce memory footprint by up to 60%.
        for col in df.select_dtypes(include=['int64', 'int32']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        return df

class ChunkedStreamProcessor:
    # Stream large datasets in chunks to handle multi-gigabyte files efficiently.

    def __init__(self, filepath: str, chunk_size: int = 10000):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.ext = os.path.splitext(filepath)[1].lower()

    def stream_chunks(self) -> Generator[pd.DataFrame, None, None]:
        # Yield chunks of Pandas DataFrames.
        if self.ext == ".csv":
            for chunk in pd.read_csv(self.filepath, chunksize=self.chunk_size):
                yield DataLoader.optimize_dtypes(chunk)
        elif self.ext == ".parquet":
            parquet_file = pq.ParquetFile(self.filepath)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                yield DataLoader.optimize_dtypes(batch.to_pandas())
        else:
            df = DataLoader.load_dataframe(self.filepath)
            for i in range(0, len(df), self.chunk_size):
                yield df.iloc[i:i + self.chunk_size]
