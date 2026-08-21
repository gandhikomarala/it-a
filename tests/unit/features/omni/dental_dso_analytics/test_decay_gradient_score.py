# Unit Test for DecayGradientScoreExtractor_Dentaldsoanalytics (Dental DSO Practice Optimization).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.dental_dso_analytics.decay_gradient_score import DecayGradientScoreExtractor_Dentaldsoanalytics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_dental_dso_analytics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Dentaldsoanalytics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_dental_dso_analytics_signal" in res.columns
    assert f"decay_gradient_score_dental_dso_analytics_risk_score" in res.columns
    assert not res[f"decay_gradient_score_dental_dso_analytics_signal"].isnull().any()

def test_decay_gradient_score_dental_dso_analytics_empty():
    extractor = DecayGradientScoreExtractor_Dentaldsoanalytics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
