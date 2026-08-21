# Unit Test for DataThrottleLimitHits (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.data_throttle_limit_hits import DataThrottleLimitHits
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_data_throttle_limit_hits_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DataThrottleLimitHits()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"data_throttle_limit_hits_signal" in res.columns
    assert f"data_throttle_limit_hits_risk_index" in res.columns
    assert not res[f"data_throttle_limit_hits_signal"].isnull().any()

def test_data_throttle_limit_hits_empty_handling():
    extractor = DataThrottleLimitHits()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
