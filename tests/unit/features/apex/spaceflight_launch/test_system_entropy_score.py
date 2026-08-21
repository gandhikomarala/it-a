# Unit Test for SystemEntropyScoreExtractor_Spaceflightlaunch (Commercial Space Flight Launch Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.spaceflight_launch.system_entropy_score import SystemEntropyScoreExtractor_Spaceflightlaunch
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_spaceflight_launch_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Spaceflightlaunch()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_spaceflight_launch_signal" in res.columns
    assert f"system_entropy_score_spaceflight_launch_risk_score" in res.columns
    assert not res[f"system_entropy_score_spaceflight_launch_signal"].isnull().any()

def test_system_entropy_score_spaceflight_launch_empty():
    extractor = SystemEntropyScoreExtractor_Spaceflightlaunch()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
