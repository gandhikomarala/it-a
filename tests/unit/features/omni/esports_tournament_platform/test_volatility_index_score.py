# Unit Test for VolatilityIndexScoreExtractor_Esportstournamentplatform (Esports Tournament & Streaming Platform).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.esports_tournament_platform.volatility_index_score import VolatilityIndexScoreExtractor_Esportstournamentplatform
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_esports_tournament_platform_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Esportstournamentplatform()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_esports_tournament_platform_signal" in res.columns
    assert f"volatility_index_score_esports_tournament_platform_risk_score" in res.columns
    assert not res[f"volatility_index_score_esports_tournament_platform_signal"].isnull().any()

def test_volatility_index_score_esports_tournament_platform_empty():
    extractor = VolatilityIndexScoreExtractor_Esportstournamentplatform()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
