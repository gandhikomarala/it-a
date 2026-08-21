# Comprehensive Unit Test for PushNotificationCTRDecayExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.push_notification_ctr_decay import PushNotificationCTRDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_push_notification_ctr_decay_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PushNotificationCTRDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"push_notification_ctr_decay_signal" in res.columns
    assert f"push_notification_ctr_decay_risk_score" in res.columns
    assert not res[f"push_notification_ctr_decay_signal"].isnull().any()

def test_push_notification_ctr_decay_empty():
    extractor = PushNotificationCTRDecayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
