# Unit Test for TransientVoltageOvershootExtractor_Cryogenicsuperconductingcavity (Superconducting Particle Accelerator RF).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.cryogenic_superconducting_cavity.transient_voltage_overshoot import TransientVoltageOvershootExtractor_Cryogenicsuperconductingcavity
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_transient_voltage_overshoot_cryogenic_superconducting_cavity_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TransientVoltageOvershootExtractor_Cryogenicsuperconductingcavity()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"transient_voltage_overshoot_cryogenic_superconducting_cavity_signal" in res.columns
    assert f"transient_voltage_overshoot_cryogenic_superconducting_cavity_risk_score" in res.columns
    assert not res[f"transient_voltage_overshoot_cryogenic_superconducting_cavity_signal"].isnull().any()

def test_transient_voltage_overshoot_cryogenic_superconducting_cavity_empty():
    extractor = TransientVoltageOvershootExtractor_Cryogenicsuperconductingcavity()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
