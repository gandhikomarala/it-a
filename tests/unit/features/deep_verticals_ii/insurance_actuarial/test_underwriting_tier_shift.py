# Comprehensive Unit Test for UnderwritingTierShiftExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.underwriting_tier_shift import UnderwritingTierShiftExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_underwriting_tier_shift_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UnderwritingTierShiftExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"underwriting_tier_shift_signal" in res.columns
    assert f"underwriting_tier_shift_risk_score" in res.columns
    assert not res[f"underwriting_tier_shift_signal"].isnull().any()

def test_underwriting_tier_shift_empty_handling():
    extractor = UnderwritingTierShiftExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
