# Unit Test for EngagementMomentumExtractor_Esportstournamentplatform (Esports Tournament & Streaming Platform).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.esports_tournament_platform.engagement_momentum import EngagementMomentumExtractor_Esportstournamentplatform
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_esports_tournament_platform_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Esportstournamentplatform()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_esports_tournament_platform_signal" in res.columns
    assert f"engagement_momentum_esports_tournament_platform_risk_score" in res.columns
    assert not res[f"engagement_momentum_esports_tournament_platform_signal"].isnull().any()

def test_engagement_momentum_esports_tournament_platform_empty():
    extractor = EngagementMomentumExtractor_Esportstournamentplatform()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
