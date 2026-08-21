# Comprehensive Unit Test for OfflineDownloadSyncRateExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.offline_download_sync_rate import OfflineDownloadSyncRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_offline_download_sync_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OfflineDownloadSyncRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"offline_download_sync_rate_signal" in res.columns
    assert f"offline_download_sync_rate_risk_score" in res.columns
    assert not res[f"offline_download_sync_rate_signal"].isnull().any()

def test_offline_download_sync_rate_empty_handling():
    extractor = OfflineDownloadSyncRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
