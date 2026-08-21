# Unit Test for EngagementMomentumExtractor_Optometryvision (Optometry & Optical Retail Chain).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.optometry_vision.engagement_momentum import EngagementMomentumExtractor_Optometryvision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_optometry_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Optometryvision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_optometry_vision_signal" in res.columns
    assert f"engagement_momentum_optometry_vision_risk_score" in res.columns
    assert not res[f"engagement_momentum_optometry_vision_signal"].isnull().any()

def test_engagement_momentum_optometry_vision_empty():
    extractor = EngagementMomentumExtractor_Optometryvision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
