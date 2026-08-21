# Comprehensive Unit Test for BillShockVarianceRatioExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.bill_shock_variance_ratio import BillShockVarianceRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_bill_shock_variance_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BillShockVarianceRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"bill_shock_variance_ratio_signal" in res.columns
    assert f"bill_shock_variance_ratio_risk_score" in res.columns
    assert not res[f"bill_shock_variance_ratio_signal"].isnull().any()

def test_bill_shock_variance_ratio_empty_handling():
    extractor = BillShockVarianceRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
