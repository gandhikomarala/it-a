# Unit Test for FeatureAdoptionBreadthExtractor_Sportsanalyticsteams (Professional Sports Team Athlete Tracking).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.sports_analytics_teams.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Sportsanalyticsteams
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_sports_analytics_teams_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Sportsanalyticsteams()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_sports_analytics_teams_signal" in res.columns
    assert f"feature_adoption_breadth_sports_analytics_teams_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_sports_analytics_teams_signal"].isnull().any()

def test_feature_adoption_breadth_sports_analytics_teams_empty():
    extractor = FeatureAdoptionBreadthExtractor_Sportsanalyticsteams()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
