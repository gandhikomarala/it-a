# Unit Test for UsageIntensityRatioExtractor_Sportsanalyticsteams (Professional Sports Team Athlete Tracking).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.sports_analytics_teams.usage_intensity_ratio import UsageIntensityRatioExtractor_Sportsanalyticsteams
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_sports_analytics_teams_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Sportsanalyticsteams()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_sports_analytics_teams_signal" in res.columns
    assert f"usage_intensity_ratio_sports_analytics_teams_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_sports_analytics_teams_signal"].isnull().any()

def test_usage_intensity_ratio_sports_analytics_teams_empty():
    extractor = UsageIntensityRatioExtractor_Sportsanalyticsteams()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
