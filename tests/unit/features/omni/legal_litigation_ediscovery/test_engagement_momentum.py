# Unit Test for EngagementMomentumExtractor_Legallitigationediscovery (Complex Litigation E-Discovery Review).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.legal_litigation_ediscovery.engagement_momentum import EngagementMomentumExtractor_Legallitigationediscovery
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_legal_litigation_ediscovery_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Legallitigationediscovery()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_legal_litigation_ediscovery_signal" in res.columns
    assert f"engagement_momentum_legal_litigation_ediscovery_risk_score" in res.columns
    assert not res[f"engagement_momentum_legal_litigation_ediscovery_signal"].isnull().any()

def test_engagement_momentum_legal_litigation_ediscovery_empty():
    extractor = EngagementMomentumExtractor_Legallitigationediscovery()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
