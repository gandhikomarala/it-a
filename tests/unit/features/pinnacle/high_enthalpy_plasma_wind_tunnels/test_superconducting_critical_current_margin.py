# Unit Test for CriticalCurrentMarginExtractor_Highenthalpyplasmawindtunnels (Arc-Jet Reentry Thermal Protection Testing).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.high_enthalpy_plasma_wind_tunnels.superconducting_critical_current_margin import CriticalCurrentMarginExtractor_Highenthalpyplasmawindtunnels
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_superconducting_critical_current_margin_high_enthalpy_plasma_wind_tunnels_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalCurrentMarginExtractor_Highenthalpyplasmawindtunnels()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"superconducting_critical_current_margin_high_enthalpy_plasma_wind_tunnels_signal" in res.columns
    assert f"superconducting_critical_current_margin_high_enthalpy_plasma_wind_tunnels_risk_score" in res.columns
    assert not res[f"superconducting_critical_current_margin_high_enthalpy_plasma_wind_tunnels_signal"].isnull().any()

def test_superconducting_critical_current_margin_high_enthalpy_plasma_wind_tunnels_empty():
    extractor = CriticalCurrentMarginExtractor_Highenthalpyplasmawindtunnels()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
