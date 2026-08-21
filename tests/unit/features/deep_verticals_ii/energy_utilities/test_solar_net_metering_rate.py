# Comprehensive Unit Test for SolarNetMeteringRateExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.solar_net_metering_rate import SolarNetMeteringRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_solar_net_metering_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SolarNetMeteringRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"solar_net_metering_rate_signal" in res.columns
    assert f"solar_net_metering_rate_risk_score" in res.columns
    assert not res[f"solar_net_metering_rate_signal"].isnull().any()

def test_solar_net_metering_rate_empty_handling():
    extractor = SolarNetMeteringRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
