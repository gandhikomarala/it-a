# Unit Test for CriticalToleranceBreachExtractor_Evfleetcharging (Commercial Electric Vehicle Fleet Charging).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.ev_fleet_charging.critical_tolerance_breach import CriticalToleranceBreachExtractor_Evfleetcharging
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_ev_fleet_charging_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Evfleetcharging()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_ev_fleet_charging_signal" in res.columns
    assert f"critical_tolerance_breach_ev_fleet_charging_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_ev_fleet_charging_signal"].isnull().any()

def test_critical_tolerance_breach_ev_fleet_charging_empty():
    extractor = CriticalToleranceBreachExtractor_Evfleetcharging()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
