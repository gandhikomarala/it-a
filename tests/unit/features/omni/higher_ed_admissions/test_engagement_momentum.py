# Unit Test for EngagementMomentumExtractor_Higheredadmissions (University Admissions & Enrollment).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.higher_ed_admissions.engagement_momentum import EngagementMomentumExtractor_Higheredadmissions
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_higher_ed_admissions_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Higheredadmissions()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_higher_ed_admissions_signal" in res.columns
    assert f"engagement_momentum_higher_ed_admissions_risk_score" in res.columns
    assert not res[f"engagement_momentum_higher_ed_admissions_signal"].isnull().any()

def test_engagement_momentum_higher_ed_admissions_empty():
    extractor = EngagementMomentumExtractor_Higheredadmissions()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
