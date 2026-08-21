# Unit Test for CriticalToleranceBreachExtractor_Directedenergyoptical (High-Energy Laser Beam Control Systems).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.directed_energy_optical.critical_tolerance_breach import CriticalToleranceBreachExtractor_Directedenergyoptical
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_directed_energy_optical_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Directedenergyoptical()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_directed_energy_optical_signal" in res.columns
    assert f"critical_tolerance_breach_directed_energy_optical_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_directed_energy_optical_signal"].isnull().any()

def test_critical_tolerance_breach_directed_energy_optical_empty():
    extractor = CriticalToleranceBreachExtractor_Directedenergyoptical()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
