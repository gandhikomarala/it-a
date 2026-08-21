# Unit Test for SystemEntropyScoreExtractor_Geothermalenergydeep (Enhanced Geothermal Deep Drilling Systems).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.geothermal_energy_deep.system_entropy_score import SystemEntropyScoreExtractor_Geothermalenergydeep
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_geothermal_energy_deep_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Geothermalenergydeep()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_geothermal_energy_deep_signal" in res.columns
    assert f"system_entropy_score_geothermal_energy_deep_risk_score" in res.columns
    assert not res[f"system_entropy_score_geothermal_energy_deep_signal"].isnull().any()

def test_system_entropy_score_geothermal_energy_deep_empty():
    extractor = SystemEntropyScoreExtractor_Geothermalenergydeep()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
