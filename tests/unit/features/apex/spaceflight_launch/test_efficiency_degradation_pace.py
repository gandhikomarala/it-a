# Unit Test for EfficiencyDegradationPaceExtractor_Spaceflightlaunch (Commercial Space Flight Launch Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.spaceflight_launch.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Spaceflightlaunch
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_spaceflight_launch_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Spaceflightlaunch()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_spaceflight_launch_signal" in res.columns
    assert f"efficiency_degradation_pace_spaceflight_launch_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_spaceflight_launch_signal"].isnull().any()

def test_efficiency_degradation_pace_spaceflight_launch_empty():
    extractor = EfficiencyDegradationPaceExtractor_Spaceflightlaunch()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
