# Comprehensive Unit Test for DealershipServiceRetentionExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.dealership_service_retention_flag import DealershipServiceRetentionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_dealership_service_retention_flag_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DealershipServiceRetentionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"dealership_service_retention_flag_signal" in res.columns
    assert f"dealership_service_retention_flag_risk_score" in res.columns
    assert not res[f"dealership_service_retention_flag_signal"].isnull().any()

def test_dealership_service_retention_flag_empty_handling():
    extractor = DealershipServiceRetentionExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
