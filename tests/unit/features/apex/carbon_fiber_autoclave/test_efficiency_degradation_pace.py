# Unit Test for EfficiencyDegradationPaceExtractor_Carbonfiberautoclave (Aerospace Carbon Fiber Composite Autoclaves).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.carbon_fiber_autoclave.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Carbonfiberautoclave
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_carbon_fiber_autoclave_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Carbonfiberautoclave()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_carbon_fiber_autoclave_signal" in res.columns
    assert f"efficiency_degradation_pace_carbon_fiber_autoclave_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_carbon_fiber_autoclave_signal"].isnull().any()

def test_efficiency_degradation_pace_carbon_fiber_autoclave_empty():
    extractor = EfficiencyDegradationPaceExtractor_Carbonfiberautoclave()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
