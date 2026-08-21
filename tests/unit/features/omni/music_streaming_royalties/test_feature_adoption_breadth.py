# Unit Test for FeatureAdoptionBreadthExtractor_Musicstreamingroyalties (Music Streaming Artist Royalty Split).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.music_streaming_royalties.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Musicstreamingroyalties
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_music_streaming_royalties_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Musicstreamingroyalties()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_music_streaming_royalties_signal" in res.columns
    assert f"feature_adoption_breadth_music_streaming_royalties_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_music_streaming_royalties_signal"].isnull().any()

def test_feature_adoption_breadth_music_streaming_royalties_empty():
    extractor = FeatureAdoptionBreadthExtractor_Musicstreamingroyalties()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
