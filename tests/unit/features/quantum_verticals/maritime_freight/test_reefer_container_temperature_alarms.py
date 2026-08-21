# Comprehensive Unit Test for ReeferTempAlarmsExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.reefer_container_temperature_alarms import ReeferTempAlarmsExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_reefer_container_temperature_alarms_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ReeferTempAlarmsExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"reefer_container_temperature_alarms_signal" in res.columns
    assert f"reefer_container_temperature_alarms_risk_score" in res.columns
    assert not res[f"reefer_container_temperature_alarms_signal"].isnull().any()

def test_reefer_container_temperature_alarms_empty_handling():
    extractor = ReeferTempAlarmsExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
