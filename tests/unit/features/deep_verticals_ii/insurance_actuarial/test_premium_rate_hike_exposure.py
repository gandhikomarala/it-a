# Comprehensive Unit Test for PremiumRateHikeExposureExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.premium_rate_hike_exposure import PremiumRateHikeExposureExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_premium_rate_hike_exposure_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PremiumRateHikeExposureExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"premium_rate_hike_exposure_signal" in res.columns
    assert f"premium_rate_hike_exposure_risk_score" in res.columns
    assert not res[f"premium_rate_hike_exposure_signal"].isnull().any()

def test_premium_rate_hike_exposure_empty_handling():
    extractor = PremiumRateHikeExposureExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
