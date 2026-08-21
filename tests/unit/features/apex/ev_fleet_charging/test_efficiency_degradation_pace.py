# Unit Test for EfficiencyDegradationPaceExtractor_Evfleetcharging (Commercial Electric Vehicle Fleet Charging).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.ev_fleet_charging.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Evfleetcharging
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_ev_fleet_charging_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Evfleetcharging()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_ev_fleet_charging_signal" in res.columns
    assert f"efficiency_degradation_pace_ev_fleet_charging_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_ev_fleet_charging_signal"].isnull().any()

def test_efficiency_degradation_pace_ev_fleet_charging_empty():
    extractor = EfficiencyDegradationPaceExtractor_Evfleetcharging()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
