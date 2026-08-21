# Unit Test for EngagementMomentumExtractor_Reinsurancecatastrophe (Catastrophe Reinsurance Modeling).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.reinsurance_catastrophe.engagement_momentum import EngagementMomentumExtractor_Reinsurancecatastrophe
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_reinsurance_catastrophe_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Reinsurancecatastrophe()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_reinsurance_catastrophe_signal" in res.columns
    assert f"engagement_momentum_reinsurance_catastrophe_risk_score" in res.columns
    assert not res[f"engagement_momentum_reinsurance_catastrophe_signal"].isnull().any()

def test_engagement_momentum_reinsurance_catastrophe_empty():
    extractor = EngagementMomentumExtractor_Reinsurancecatastrophe()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
