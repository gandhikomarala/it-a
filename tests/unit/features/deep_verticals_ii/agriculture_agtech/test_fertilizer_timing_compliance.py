# Comprehensive Unit Test for FertilizerComplianceExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.fertilizer_timing_compliance import FertilizerComplianceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_fertilizer_timing_compliance_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FertilizerComplianceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"fertilizer_timing_compliance_signal" in res.columns
    assert f"fertilizer_timing_compliance_risk_score" in res.columns
    assert not res[f"fertilizer_timing_compliance_signal"].isnull().any()

def test_fertilizer_timing_compliance_empty_handling():
    extractor = FertilizerComplianceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
