# Comprehensive Unit Test for UnsubscribeVelocityExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.unsubscription_velocity_spike import UnsubscribeVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_unsubscription_velocity_spike_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UnsubscribeVelocityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"unsubscription_velocity_spike_signal" in res.columns
    assert f"unsubscription_velocity_spike_risk_score" in res.columns
    assert not res[f"unsubscription_velocity_spike_signal"].isnull().any()

def test_unsubscription_velocity_spike_empty():
    extractor = UnsubscribeVelocityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
