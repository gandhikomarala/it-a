# Unit Test for OverdraftFrequencySurgeExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.overdraft_frequency_surge import OverdraftFrequencySurgeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_overdraft_frequency_surge_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OverdraftFrequencySurgeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"overdraft_frequency_surge_signal" in res.columns
    assert f"overdraft_frequency_surge_risk_score" in res.columns
    assert not res[f"overdraft_frequency_surge_signal"].isnull().any()

def test_overdraft_frequency_surge_empty_dataframe():
    extractor = OverdraftFrequencySurgeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
