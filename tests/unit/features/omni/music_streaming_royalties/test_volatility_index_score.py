# Unit Test for VolatilityIndexScoreExtractor_Musicstreamingroyalties (Music Streaming Artist Royalty Split).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.music_streaming_royalties.volatility_index_score import VolatilityIndexScoreExtractor_Musicstreamingroyalties
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_music_streaming_royalties_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Musicstreamingroyalties()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_music_streaming_royalties_signal" in res.columns
    assert f"volatility_index_score_music_streaming_royalties_risk_score" in res.columns
    assert not res[f"volatility_index_score_music_streaming_royalties_signal"].isnull().any()

def test_volatility_index_score_music_streaming_royalties_empty():
    extractor = VolatilityIndexScoreExtractor_Musicstreamingroyalties()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
