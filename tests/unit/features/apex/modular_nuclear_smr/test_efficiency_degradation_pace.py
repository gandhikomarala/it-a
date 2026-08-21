# Unit Test for EfficiencyDegradationPaceExtractor_Modularnuclearsmr (Small Modular Nuclear Reactor (SMR) Systems).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.modular_nuclear_smr.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Modularnuclearsmr
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_modular_nuclear_smr_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Modularnuclearsmr()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_modular_nuclear_smr_signal" in res.columns
    assert f"efficiency_degradation_pace_modular_nuclear_smr_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_modular_nuclear_smr_signal"].isnull().any()

def test_efficiency_degradation_pace_modular_nuclear_smr_empty():
    extractor = EfficiencyDegradationPaceExtractor_Modularnuclearsmr()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
