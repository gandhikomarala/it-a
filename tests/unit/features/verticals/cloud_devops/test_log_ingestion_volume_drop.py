# Unit Test for LogIngestionVolumeDropExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.log_ingestion_volume_drop import LogIngestionVolumeDropExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_log_ingestion_volume_drop_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LogIngestionVolumeDropExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"log_ingestion_volume_drop_signal" in res.columns
    assert f"log_ingestion_volume_drop_risk_score" in res.columns
    assert not res[f"log_ingestion_volume_drop_signal"].isnull().any()

def test_log_ingestion_volume_drop_empty_dataframe():
    extractor = LogIngestionVolumeDropExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
