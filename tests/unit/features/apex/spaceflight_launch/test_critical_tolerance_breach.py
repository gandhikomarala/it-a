# Unit Test for CriticalToleranceBreachExtractor_Spaceflightlaunch (Commercial Space Flight Launch Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.spaceflight_launch.critical_tolerance_breach import CriticalToleranceBreachExtractor_Spaceflightlaunch
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_spaceflight_launch_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Spaceflightlaunch()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_spaceflight_launch_signal" in res.columns
    assert f"critical_tolerance_breach_spaceflight_launch_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_spaceflight_launch_signal"].isnull().any()

def test_critical_tolerance_breach_spaceflight_launch_empty():
    extractor = CriticalToleranceBreachExtractor_Spaceflightlaunch()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
