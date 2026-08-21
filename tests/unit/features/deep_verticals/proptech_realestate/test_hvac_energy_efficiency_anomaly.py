# Comprehensive Unit Test for HVACEnergyAnomalyExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.hvac_energy_efficiency_anomaly import HVACEnergyAnomalyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_hvac_energy_efficiency_anomaly_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HVACEnergyAnomalyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"hvac_energy_efficiency_anomaly_signal" in res.columns
    assert f"hvac_energy_efficiency_anomaly_risk_score" in res.columns
    assert not res[f"hvac_energy_efficiency_anomaly_signal"].isnull().any()

def test_hvac_energy_efficiency_anomaly_empty():
    extractor = HVACEnergyAnomalyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
