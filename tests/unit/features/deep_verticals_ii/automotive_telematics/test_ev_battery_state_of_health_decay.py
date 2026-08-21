# Comprehensive Unit Test for EVBatterySOHDecayExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.ev_battery_state_of_health_decay import EVBatterySOHDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ev_battery_state_of_health_decay_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EVBatterySOHDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ev_battery_state_of_health_decay_signal" in res.columns
    assert f"ev_battery_state_of_health_decay_risk_score" in res.columns
    assert not res[f"ev_battery_state_of_health_decay_signal"].isnull().any()

def test_ev_battery_state_of_health_decay_empty_handling():
    extractor = EVBatterySOHDecayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
