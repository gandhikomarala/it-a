# Comprehensive Unit Test for BunkerFuelAnomalyExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.bunker_fuel_consumption_anomaly import BunkerFuelAnomalyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_bunker_fuel_consumption_anomaly_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BunkerFuelAnomalyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"bunker_fuel_consumption_anomaly_signal" in res.columns
    assert f"bunker_fuel_consumption_anomaly_risk_score" in res.columns
    assert not res[f"bunker_fuel_consumption_anomaly_signal"].isnull().any()

def test_bunker_fuel_consumption_anomaly_empty_handling():
    extractor = BunkerFuelAnomalyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
