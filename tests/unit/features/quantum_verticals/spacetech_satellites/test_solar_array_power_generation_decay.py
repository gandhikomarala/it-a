# Comprehensive Unit Test for SolarArrayPowerDecayExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.solar_array_power_generation_decay import SolarArrayPowerDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_solar_array_power_generation_decay_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SolarArrayPowerDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"solar_array_power_generation_decay_signal" in res.columns
    assert f"solar_array_power_generation_decay_risk_score" in res.columns
    assert not res[f"solar_array_power_generation_decay_signal"].isnull().any()

def test_solar_array_power_generation_decay_empty_handling():
    extractor = SolarArrayPowerDecayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
