# Comprehensive Unit Test for NetworkEnergyEfficiencyKPIExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.network_energy_efficiency_kpi import NetworkEnergyEfficiencyKPIExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_network_energy_efficiency_kpi_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = NetworkEnergyEfficiencyKPIExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"network_energy_efficiency_kpi_signal" in res.columns
    assert f"network_energy_efficiency_kpi_risk_score" in res.columns
    assert not res[f"network_energy_efficiency_kpi_signal"].isnull().any()

def test_network_energy_efficiency_kpi_empty_handling():
    extractor = NetworkEnergyEfficiencyKPIExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
