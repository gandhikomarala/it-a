# Unit Test for CriticalCurrentMarginExtractor_Cryogenicsuperconductingcavity (Superconducting Particle Accelerator RF).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.cryogenic_superconducting_cavity.superconducting_critical_current_margin import CriticalCurrentMarginExtractor_Cryogenicsuperconductingcavity
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_superconducting_critical_current_margin_cryogenic_superconducting_cavity_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalCurrentMarginExtractor_Cryogenicsuperconductingcavity()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"superconducting_critical_current_margin_cryogenic_superconducting_cavity_signal" in res.columns
    assert f"superconducting_critical_current_margin_cryogenic_superconducting_cavity_risk_score" in res.columns
    assert not res[f"superconducting_critical_current_margin_cryogenic_superconducting_cavity_signal"].isnull().any()

def test_superconducting_critical_current_margin_cryogenic_superconducting_cavity_empty():
    extractor = CriticalCurrentMarginExtractor_Cryogenicsuperconductingcavity()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
