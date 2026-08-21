# Comprehensive Unit Test for IrrigationSensorUptimeExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.irrigation_sensor_uptime_pct import IrrigationSensorUptimeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_irrigation_sensor_uptime_pct_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = IrrigationSensorUptimeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"irrigation_sensor_uptime_pct_signal" in res.columns
    assert f"irrigation_sensor_uptime_pct_risk_score" in res.columns
    assert not res[f"irrigation_sensor_uptime_pct_signal"].isnull().any()

def test_irrigation_sensor_uptime_pct_empty_handling():
    extractor = IrrigationSensorUptimeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
