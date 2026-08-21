# Comprehensive Unit Test for OEEAvailabilityDropExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.oee_availability_drop_pct import OEEAvailabilityDropExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_oee_availability_drop_pct_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OEEAvailabilityDropExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"oee_availability_drop_pct_signal" in res.columns
    assert f"oee_availability_drop_pct_risk_score" in res.columns
    assert not res[f"oee_availability_drop_pct_signal"].isnull().any()

def test_oee_availability_drop_pct_empty_handling():
    extractor = OEEAvailabilityDropExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
