# Unit Test for EfficiencyDegradationPaceExtractor_Superconductingmagnetsmri (Ultra-High Field 7T MRI Superconducting Magnets).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.superconducting_magnets_mri.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Superconductingmagnetsmri
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_superconducting_magnets_mri_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Superconductingmagnetsmri()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_superconducting_magnets_mri_signal" in res.columns
    assert f"efficiency_degradation_pace_superconducting_magnets_mri_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_superconducting_magnets_mri_signal"].isnull().any()

def test_efficiency_degradation_pace_superconducting_magnets_mri_empty():
    extractor = EfficiencyDegradationPaceExtractor_Superconductingmagnetsmri()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
