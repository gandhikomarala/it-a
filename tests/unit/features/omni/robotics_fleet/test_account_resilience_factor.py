# Unit Test for AccountResilienceFactorExtractor_Roboticsfleet (Autonomous Robotics & AMR Fleet Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.robotics_fleet.account_resilience_factor import AccountResilienceFactorExtractor_Roboticsfleet
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_robotics_fleet_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Roboticsfleet()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_robotics_fleet_signal" in res.columns
    assert f"account_resilience_factor_robotics_fleet_risk_score" in res.columns
    assert not res[f"account_resilience_factor_robotics_fleet_signal"].isnull().any()

def test_account_resilience_factor_robotics_fleet_empty():
    extractor = AccountResilienceFactorExtractor_Roboticsfleet()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
