# Unit Test for EngagementMomentumExtractor_Carbonaccountingesg (Enterprise Scope 1-2-3 Carbon Accounting).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.carbon_accounting_esg.engagement_momentum import EngagementMomentumExtractor_Carbonaccountingesg
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_carbon_accounting_esg_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Carbonaccountingesg()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_carbon_accounting_esg_signal" in res.columns
    assert f"engagement_momentum_carbon_accounting_esg_risk_score" in res.columns
    assert not res[f"engagement_momentum_carbon_accounting_esg_signal"].isnull().any()

def test_engagement_momentum_carbon_accounting_esg_empty():
    extractor = EngagementMomentumExtractor_Carbonaccountingesg()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
