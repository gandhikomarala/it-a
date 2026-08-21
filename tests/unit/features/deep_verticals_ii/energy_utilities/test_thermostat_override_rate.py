# Comprehensive Unit Test for ThermostatOverrideRateExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.thermostat_override_rate import ThermostatOverrideRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_thermostat_override_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ThermostatOverrideRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"thermostat_override_rate_signal" in res.columns
    assert f"thermostat_override_rate_risk_score" in res.columns
    assert not res[f"thermostat_override_rate_signal"].isnull().any()

def test_thermostat_override_rate_empty_handling():
    extractor = ThermostatOverrideRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
