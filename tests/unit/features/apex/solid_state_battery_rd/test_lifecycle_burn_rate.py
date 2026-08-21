# Unit Test for LifecycleBurnRateExtractor_Solidstatebatteryrd (All-Solid-State Battery Anode Electrolyte).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.solid_state_battery_rd.lifecycle_burn_rate import LifecycleBurnRateExtractor_Solidstatebatteryrd
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_solid_state_battery_rd_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Solidstatebatteryrd()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_solid_state_battery_rd_signal" in res.columns
    assert f"lifecycle_burn_rate_solid_state_battery_rd_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_solid_state_battery_rd_signal"].isnull().any()

def test_lifecycle_burn_rate_solid_state_battery_rd_empty():
    extractor = LifecycleBurnRateExtractor_Solidstatebatteryrd()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
