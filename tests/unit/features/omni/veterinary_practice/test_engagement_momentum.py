# Unit Test for EngagementMomentumExtractor_Veterinarypractice (Veterinary Practice Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.veterinary_practice.engagement_momentum import EngagementMomentumExtractor_Veterinarypractice
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_veterinary_practice_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Veterinarypractice()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_veterinary_practice_signal" in res.columns
    assert f"engagement_momentum_veterinary_practice_risk_score" in res.columns
    assert not res[f"engagement_momentum_veterinary_practice_signal"].isnull().any()

def test_engagement_momentum_veterinary_practice_empty():
    extractor = EngagementMomentumExtractor_Veterinarypractice()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
