# Unit Test for LifecycleBurnRateExtractor_Carbonfiberautoclave (Aerospace Carbon Fiber Composite Autoclaves).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.carbon_fiber_autoclave.lifecycle_burn_rate import LifecycleBurnRateExtractor_Carbonfiberautoclave
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_carbon_fiber_autoclave_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Carbonfiberautoclave()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_carbon_fiber_autoclave_signal" in res.columns
    assert f"lifecycle_burn_rate_carbon_fiber_autoclave_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_carbon_fiber_autoclave_signal"].isnull().any()

def test_lifecycle_burn_rate_carbon_fiber_autoclave_empty():
    extractor = LifecycleBurnRateExtractor_Carbonfiberautoclave()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
