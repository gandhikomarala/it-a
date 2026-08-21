# Unit Test for EfficiencyDegradationPaceExtractor_Nuclearenergyiot (Nuclear Energy & Power Plant IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.nuclear_energy_iot.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Nuclearenergyiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_nuclear_energy_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Nuclearenergyiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_nuclear_energy_iot_signal" in res.columns
    assert f"efficiency_degradation_pace_nuclear_energy_iot_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_nuclear_energy_iot_signal"].isnull().any()

def test_efficiency_degradation_pace_nuclear_energy_iot_empty():
    extractor = EfficiencyDegradationPaceExtractor_Nuclearenergyiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
