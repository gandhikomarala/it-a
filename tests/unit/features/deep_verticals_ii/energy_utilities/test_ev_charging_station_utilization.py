# Comprehensive Unit Test for EVChargingUtilizationExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.ev_charging_station_utilization import EVChargingUtilizationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ev_charging_station_utilization_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EVChargingUtilizationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ev_charging_station_utilization_signal" in res.columns
    assert f"ev_charging_station_utilization_risk_score" in res.columns
    assert not res[f"ev_charging_station_utilization_signal"].isnull().any()

def test_ev_charging_station_utilization_empty_handling():
    extractor = EVChargingUtilizationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
