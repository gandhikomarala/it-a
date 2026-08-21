# Unit Test for EfficiencyDegradationPaceExtractor_Desalinationwater (Reverse Osmosis Sea Water Desalination).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.desalination_water.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Desalinationwater
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_desalination_water_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Desalinationwater()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_desalination_water_signal" in res.columns
    assert f"efficiency_degradation_pace_desalination_water_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_desalination_water_signal"].isnull().any()

def test_efficiency_degradation_pace_desalination_water_empty():
    extractor = EfficiencyDegradationPaceExtractor_Desalinationwater()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
