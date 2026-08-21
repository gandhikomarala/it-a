# Unit Test for DecayGradientScoreExtractor_Wealthadvisory (Private Wealth Advisory & Estate Planning).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.wealth_advisory.decay_gradient_score import DecayGradientScoreExtractor_Wealthadvisory
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_wealth_advisory_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Wealthadvisory()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_wealth_advisory_signal" in res.columns
    assert f"decay_gradient_score_wealth_advisory_risk_score" in res.columns
    assert not res[f"decay_gradient_score_wealth_advisory_signal"].isnull().any()

def test_decay_gradient_score_wealth_advisory_empty():
    extractor = DecayGradientScoreExtractor_Wealthadvisory()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
