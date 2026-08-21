# Unit Test for EfficiencyDegradationPaceExtractor_Spacedebrislasertracking (Ground-Based Laser Space Debris Tracking).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.space_debris_laser_tracking.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Spacedebrislasertracking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_space_debris_laser_tracking_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Spacedebrislasertracking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_space_debris_laser_tracking_signal" in res.columns
    assert f"efficiency_degradation_pace_space_debris_laser_tracking_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_space_debris_laser_tracking_signal"].isnull().any()

def test_efficiency_degradation_pace_space_debris_laser_tracking_empty():
    extractor = EfficiencyDegradationPaceExtractor_Spacedebrislasertracking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
