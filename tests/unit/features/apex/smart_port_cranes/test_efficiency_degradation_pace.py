# Unit Test for EfficiencyDegradationPaceExtractor_Smartportcranes (Automated Container Port STS Cranes).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_port_cranes.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Smartportcranes
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_smart_port_cranes_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Smartportcranes()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_smart_port_cranes_signal" in res.columns
    assert f"efficiency_degradation_pace_smart_port_cranes_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_smart_port_cranes_signal"].isnull().any()

def test_efficiency_degradation_pace_smart_port_cranes_empty():
    extractor = EfficiencyDegradationPaceExtractor_Smartportcranes()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
