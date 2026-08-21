# Unit Test for FatigueCycleAccumulationExtractor_Cryogenicsuperconductingcavity (Superconducting Particle Accelerator RF).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.cryogenic_superconducting_cavity.structural_fatigue_cycle_accumulation import FatigueCycleAccumulationExtractor_Cryogenicsuperconductingcavity
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_structural_fatigue_cycle_accumulation_cryogenic_superconducting_cavity_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FatigueCycleAccumulationExtractor_Cryogenicsuperconductingcavity()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"structural_fatigue_cycle_accumulation_cryogenic_superconducting_cavity_signal" in res.columns
    assert f"structural_fatigue_cycle_accumulation_cryogenic_superconducting_cavity_risk_score" in res.columns
    assert not res[f"structural_fatigue_cycle_accumulation_cryogenic_superconducting_cavity_signal"].isnull().any()

def test_structural_fatigue_cycle_accumulation_cryogenic_superconducting_cavity_empty():
    extractor = FatigueCycleAccumulationExtractor_Cryogenicsuperconductingcavity()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
