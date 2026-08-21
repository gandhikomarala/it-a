# Unit Test for DecayGradientScoreExtractor_Miningheavyindustry (Mining & Heavy Equipment IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.mining_heavy_industry.decay_gradient_score import DecayGradientScoreExtractor_Miningheavyindustry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_mining_heavy_industry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Miningheavyindustry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_mining_heavy_industry_signal" in res.columns
    assert f"decay_gradient_score_mining_heavy_industry_risk_score" in res.columns
    assert not res[f"decay_gradient_score_mining_heavy_industry_signal"].isnull().any()

def test_decay_gradient_score_mining_heavy_industry_empty():
    extractor = DecayGradientScoreExtractor_Miningheavyindustry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
