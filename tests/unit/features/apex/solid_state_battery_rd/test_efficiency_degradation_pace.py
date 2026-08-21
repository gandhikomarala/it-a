# Unit Test for EfficiencyDegradationPaceExtractor_Solidstatebatteryrd (All-Solid-State Battery Anode Electrolyte).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.solid_state_battery_rd.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Solidstatebatteryrd
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_solid_state_battery_rd_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Solidstatebatteryrd()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_solid_state_battery_rd_signal" in res.columns
    assert f"efficiency_degradation_pace_solid_state_battery_rd_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_solid_state_battery_rd_signal"].isnull().any()

def test_efficiency_degradation_pace_solid_state_battery_rd_empty():
    extractor = EfficiencyDegradationPaceExtractor_Solidstatebatteryrd()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
