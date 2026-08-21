# Comprehensive Unit Test for ThermalRadiatorEfficiencyExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.thermal_radiator_efficiency_drop import ThermalRadiatorEfficiencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_thermal_radiator_efficiency_drop_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ThermalRadiatorEfficiencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"thermal_radiator_efficiency_drop_signal" in res.columns
    assert f"thermal_radiator_efficiency_drop_risk_score" in res.columns
    assert not res[f"thermal_radiator_efficiency_drop_signal"].isnull().any()

def test_thermal_radiator_efficiency_drop_empty_handling():
    extractor = ThermalRadiatorEfficiencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
