# Unit Test for MultiRegionReplicationLagExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.multi_region_replication_lag import MultiRegionReplicationLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_multi_region_replication_lag_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MultiRegionReplicationLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"multi_region_replication_lag_signal" in res.columns
    assert f"multi_region_replication_lag_risk_score" in res.columns
    assert not res[f"multi_region_replication_lag_signal"].isnull().any()

def test_multi_region_replication_lag_empty_dataframe():
    extractor = MultiRegionReplicationLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
