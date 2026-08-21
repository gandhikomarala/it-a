# Comprehensive Unit Test for PeerRecognitionCountExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.peer_shoutout_recognition_count import PeerRecognitionCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_peer_shoutout_recognition_count_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PeerRecognitionCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"peer_shoutout_recognition_count_signal" in res.columns
    assert f"peer_shoutout_recognition_count_risk_score" in res.columns
    assert not res[f"peer_shoutout_recognition_count_signal"].isnull().any()

def test_peer_shoutout_recognition_count_empty():
    extractor = PeerRecognitionCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
