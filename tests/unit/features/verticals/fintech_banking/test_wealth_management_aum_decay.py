# Unit Test for WealthManagementAUMDecayExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.wealth_management_aum_decay import WealthManagementAUMDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_wealth_management_aum_decay_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = WealthManagementAUMDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"wealth_management_aum_decay_signal" in res.columns
    assert f"wealth_management_aum_decay_risk_score" in res.columns
    assert not res[f"wealth_management_aum_decay_signal"].isnull().any()

def test_wealth_management_aum_decay_empty_dataframe():
    extractor = WealthManagementAUMDecayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
