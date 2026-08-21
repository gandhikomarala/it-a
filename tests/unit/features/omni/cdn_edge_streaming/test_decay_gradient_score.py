# Unit Test for DecayGradientScoreExtractor_Cdnedgestreaming (Global CDN Video Edge Caching).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cdn_edge_streaming.decay_gradient_score import DecayGradientScoreExtractor_Cdnedgestreaming
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_cdn_edge_streaming_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Cdnedgestreaming()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_cdn_edge_streaming_signal" in res.columns
    assert f"decay_gradient_score_cdn_edge_streaming_risk_score" in res.columns
    assert not res[f"decay_gradient_score_cdn_edge_streaming_signal"].isnull().any()

def test_decay_gradient_score_cdn_edge_streaming_empty():
    extractor = DecayGradientScoreExtractor_Cdnedgestreaming()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
