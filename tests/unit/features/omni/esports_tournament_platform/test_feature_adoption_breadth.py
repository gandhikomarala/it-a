# Unit Test for FeatureAdoptionBreadthExtractor_Esportstournamentplatform (Esports Tournament & Streaming Platform).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.esports_tournament_platform.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Esportstournamentplatform
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_esports_tournament_platform_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Esportstournamentplatform()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_esports_tournament_platform_signal" in res.columns
    assert f"feature_adoption_breadth_esports_tournament_platform_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_esports_tournament_platform_signal"].isnull().any()

def test_feature_adoption_breadth_esports_tournament_platform_empty():
    extractor = FeatureAdoptionBreadthExtractor_Esportstournamentplatform()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
