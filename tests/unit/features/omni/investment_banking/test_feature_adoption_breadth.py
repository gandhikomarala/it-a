# Unit Test for FeatureAdoptionBreadthExtractor_Investmentbanking (Investment Banking M&A Deal Pipeline).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.investment_banking.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Investmentbanking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_investment_banking_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Investmentbanking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_investment_banking_signal" in res.columns
    assert f"feature_adoption_breadth_investment_banking_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_investment_banking_signal"].isnull().any()

def test_feature_adoption_breadth_investment_banking_empty():
    extractor = FeatureAdoptionBreadthExtractor_Investmentbanking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
