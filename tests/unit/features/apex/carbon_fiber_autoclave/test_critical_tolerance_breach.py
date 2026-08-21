# Unit Test for CriticalToleranceBreachExtractor_Carbonfiberautoclave (Aerospace Carbon Fiber Composite Autoclaves).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.carbon_fiber_autoclave.critical_tolerance_breach import CriticalToleranceBreachExtractor_Carbonfiberautoclave
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_carbon_fiber_autoclave_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Carbonfiberautoclave()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_carbon_fiber_autoclave_signal" in res.columns
    assert f"critical_tolerance_breach_carbon_fiber_autoclave_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_carbon_fiber_autoclave_signal"].isnull().any()

def test_critical_tolerance_breach_carbon_fiber_autoclave_empty():
    extractor = CriticalToleranceBreachExtractor_Carbonfiberautoclave()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
