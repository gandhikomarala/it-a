# Unit Test for FeatureAdoptionBreadthExtractor_Carbonaccountingesg (Enterprise Scope 1-2-3 Carbon Accounting).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.carbon_accounting_esg.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Carbonaccountingesg
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_carbon_accounting_esg_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Carbonaccountingesg()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_carbon_accounting_esg_signal" in res.columns
    assert f"feature_adoption_breadth_carbon_accounting_esg_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_carbon_accounting_esg_signal"].isnull().any()

def test_feature_adoption_breadth_carbon_accounting_esg_empty():
    extractor = FeatureAdoptionBreadthExtractor_Carbonaccountingesg()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
