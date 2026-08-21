# Unit Test for EfficiencyDegradationPaceExtractor_Nanotechmaterialslab (Advanced Nanomaterials Synthesis Lab).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.nanotech_materials_lab.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Nanotechmaterialslab
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_nanotech_materials_lab_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Nanotechmaterialslab()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_nanotech_materials_lab_signal" in res.columns
    assert f"efficiency_degradation_pace_nanotech_materials_lab_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_nanotech_materials_lab_signal"].isnull().any()

def test_efficiency_degradation_pace_nanotech_materials_lab_empty():
    extractor = EfficiencyDegradationPaceExtractor_Nanotechmaterialslab()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
