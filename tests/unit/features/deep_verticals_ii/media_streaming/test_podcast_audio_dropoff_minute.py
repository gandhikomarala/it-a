# Comprehensive Unit Test for PodcastAudioDropoffMinuteExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.podcast_audio_dropoff_minute import PodcastAudioDropoffMinuteExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_podcast_audio_dropoff_minute_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PodcastAudioDropoffMinuteExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"podcast_audio_dropoff_minute_signal" in res.columns
    assert f"podcast_audio_dropoff_minute_risk_score" in res.columns
    assert not res[f"podcast_audio_dropoff_minute_signal"].isnull().any()

def test_podcast_audio_dropoff_minute_empty_handling():
    extractor = PodcastAudioDropoffMinuteExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
