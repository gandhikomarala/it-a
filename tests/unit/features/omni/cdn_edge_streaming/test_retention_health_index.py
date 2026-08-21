# Unit Test for RetentionHealthIndexExtractor_Cdnedgestreaming (Global CDN Video Edge Caching).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cdn_edge_streaming.retention_health_index import RetentionHealthIndexExtractor_Cdnedgestreaming
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_cdn_edge_streaming_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Cdnedgestreaming()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_cdn_edge_streaming_signal" in res.columns
    assert f"retention_health_index_cdn_edge_streaming_risk_score" in res.columns
    assert not res[f"retention_health_index_cdn_edge_streaming_signal"].isnull().any()

def test_retention_health_index_cdn_edge_streaming_empty():
    extractor = RetentionHealthIndexExtractor_Cdnedgestreaming()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
