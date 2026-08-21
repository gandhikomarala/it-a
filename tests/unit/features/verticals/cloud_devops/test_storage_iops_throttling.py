# Unit Test for StorageIOPSThrottlingExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.storage_iops_throttling import StorageIOPSThrottlingExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_storage_iops_throttling_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StorageIOPSThrottlingExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"storage_iops_throttling_signal" in res.columns
    assert f"storage_iops_throttling_risk_score" in res.columns
    assert not res[f"storage_iops_throttling_signal"].isnull().any()

def test_storage_iops_throttling_empty_dataframe():
    extractor = StorageIOPSThrottlingExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
