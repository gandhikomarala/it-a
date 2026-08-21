# Unit Test for RansomwareCanaryTripFlagExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.ransomware_canary_trip_flag import RansomwareCanaryTripFlagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ransomware_canary_trip_flag_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RansomwareCanaryTripFlagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ransomware_canary_trip_flag_signal" in res.columns
    assert f"ransomware_canary_trip_flag_risk_score" in res.columns
    assert not res[f"ransomware_canary_trip_flag_signal"].isnull().any()

def test_ransomware_canary_trip_flag_empty_dataframe():
    extractor = RansomwareCanaryTripFlagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
