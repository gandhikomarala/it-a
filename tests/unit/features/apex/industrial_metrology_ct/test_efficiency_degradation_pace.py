# Unit Test for EfficiencyDegradationPaceExtractor_Industrialmetrologyct (Industrial X-Ray Computed Tomography).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.industrial_metrology_ct.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Industrialmetrologyct
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_industrial_metrology_ct_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Industrialmetrologyct()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_industrial_metrology_ct_signal" in res.columns
    assert f"efficiency_degradation_pace_industrial_metrology_ct_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_industrial_metrology_ct_signal"].isnull().any()

def test_efficiency_degradation_pace_industrial_metrology_ct_empty():
    extractor = EfficiencyDegradationPaceExtractor_Industrialmetrologyct()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
