# Unit Test for LifecycleBurnRateExtractor_Surgicalrobotics (Precision Robotic-Assisted Surgery).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.surgical_robotics.lifecycle_burn_rate import LifecycleBurnRateExtractor_Surgicalrobotics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_surgical_robotics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Surgicalrobotics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_surgical_robotics_signal" in res.columns
    assert f"lifecycle_burn_rate_surgical_robotics_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_surgical_robotics_signal"].isnull().any()

def test_lifecycle_burn_rate_surgical_robotics_empty():
    extractor = LifecycleBurnRateExtractor_Surgicalrobotics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
