# Unit Test for DecayGradientScoreExtractor_Commercialbanking (Commercial Treasury & Syndicated Lending).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.commercial_banking.decay_gradient_score import DecayGradientScoreExtractor_Commercialbanking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_commercial_banking_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Commercialbanking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_commercial_banking_signal" in res.columns
    assert f"decay_gradient_score_commercial_banking_risk_score" in res.columns
    assert not res[f"decay_gradient_score_commercial_banking_signal"].isnull().any()

def test_decay_gradient_score_commercial_banking_empty():
    extractor = DecayGradientScoreExtractor_Commercialbanking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
