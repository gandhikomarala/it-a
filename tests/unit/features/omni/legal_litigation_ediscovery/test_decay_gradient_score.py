# Unit Test for DecayGradientScoreExtractor_Legallitigationediscovery (Complex Litigation E-Discovery Review).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.legal_litigation_ediscovery.decay_gradient_score import DecayGradientScoreExtractor_Legallitigationediscovery
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_legal_litigation_ediscovery_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Legallitigationediscovery()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_legal_litigation_ediscovery_signal" in res.columns
    assert f"decay_gradient_score_legal_litigation_ediscovery_risk_score" in res.columns
    assert not res[f"decay_gradient_score_legal_litigation_ediscovery_signal"].isnull().any()

def test_decay_gradient_score_legal_litigation_ediscovery_empty():
    extractor = DecayGradientScoreExtractor_Legallitigationediscovery()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
