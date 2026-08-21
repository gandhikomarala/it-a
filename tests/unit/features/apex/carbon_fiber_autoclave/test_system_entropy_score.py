# Unit Test for SystemEntropyScoreExtractor_Carbonfiberautoclave (Aerospace Carbon Fiber Composite Autoclaves).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.carbon_fiber_autoclave.system_entropy_score import SystemEntropyScoreExtractor_Carbonfiberautoclave
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_carbon_fiber_autoclave_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Carbonfiberautoclave()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_carbon_fiber_autoclave_signal" in res.columns
    assert f"system_entropy_score_carbon_fiber_autoclave_risk_score" in res.columns
    assert not res[f"system_entropy_score_carbon_fiber_autoclave_signal"].isnull().any()

def test_system_entropy_score_carbon_fiber_autoclave_empty():
    extractor = SystemEntropyScoreExtractor_Carbonfiberautoclave()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
