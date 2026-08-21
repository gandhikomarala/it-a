# Unit Test for EngagementMomentumExtractor_Hydroelectricdamiot (Hydroelectric Dam Structural Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.hydroelectric_dam_iot.engagement_momentum import EngagementMomentumExtractor_Hydroelectricdamiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_hydroelectric_dam_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Hydroelectricdamiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_hydroelectric_dam_iot_signal" in res.columns
    assert f"engagement_momentum_hydroelectric_dam_iot_risk_score" in res.columns
    assert not res[f"engagement_momentum_hydroelectric_dam_iot_signal"].isnull().any()

def test_engagement_momentum_hydroelectric_dam_iot_empty():
    extractor = EngagementMomentumExtractor_Hydroelectricdamiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
