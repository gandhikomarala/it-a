# Comprehensive Unit Test for DynamicPricingSensitivityExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.dynamic_pricing_sensitivity import DynamicPricingSensitivityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_dynamic_pricing_sensitivity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DynamicPricingSensitivityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"dynamic_pricing_sensitivity_signal" in res.columns
    assert f"dynamic_pricing_sensitivity_risk_score" in res.columns
    assert not res[f"dynamic_pricing_sensitivity_signal"].isnull().any()

def test_dynamic_pricing_sensitivity_empty_handling():
    extractor = DynamicPricingSensitivityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
