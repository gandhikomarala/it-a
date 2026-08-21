# Unit Test for DecayGradientScoreExtractor_Hydroelectricdamiot (Hydroelectric Dam Structural Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.hydroelectric_dam_iot.decay_gradient_score import DecayGradientScoreExtractor_Hydroelectricdamiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_hydroelectric_dam_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Hydroelectricdamiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_hydroelectric_dam_iot_signal" in res.columns
    assert f"decay_gradient_score_hydroelectric_dam_iot_risk_score" in res.columns
    assert not res[f"decay_gradient_score_hydroelectric_dam_iot_signal"].isnull().any()

def test_decay_gradient_score_hydroelectric_dam_iot_empty():
    extractor = DecayGradientScoreExtractor_Hydroelectricdamiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
