# Unit Test for TransientVoltageOvershootExtractor_Smallsatpropulsionelectrospray (Colloid Thruster Electrospray Propulsion).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.small_sat_propulsion_electrospray.transient_voltage_overshoot import TransientVoltageOvershootExtractor_Smallsatpropulsionelectrospray
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_transient_voltage_overshoot_small_sat_propulsion_electrospray_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TransientVoltageOvershootExtractor_Smallsatpropulsionelectrospray()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"transient_voltage_overshoot_small_sat_propulsion_electrospray_signal" in res.columns
    assert f"transient_voltage_overshoot_small_sat_propulsion_electrospray_risk_score" in res.columns
    assert not res[f"transient_voltage_overshoot_small_sat_propulsion_electrospray_signal"].isnull().any()

def test_transient_voltage_overshoot_small_sat_propulsion_electrospray_empty():
    extractor = TransientVoltageOvershootExtractor_Smallsatpropulsionelectrospray()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
