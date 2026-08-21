# Comprehensive Unit Test for ReactionWheelJitterExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.attitude_control_reaction_wheel_jitter import ReactionWheelJitterExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_attitude_control_reaction_wheel_jitter_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ReactionWheelJitterExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"attitude_control_reaction_wheel_jitter_signal" in res.columns
    assert f"attitude_control_reaction_wheel_jitter_risk_score" in res.columns
    assert not res[f"attitude_control_reaction_wheel_jitter_signal"].isnull().any()

def test_attitude_control_reaction_wheel_jitter_empty_handling():
    extractor = ReactionWheelJitterExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
