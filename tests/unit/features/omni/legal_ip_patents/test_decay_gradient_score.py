# Unit Test for DecayGradientScoreExtractor_Legalippatents (Patent Prosecution & IP Portfolio).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.legal_ip_patents.decay_gradient_score import DecayGradientScoreExtractor_Legalippatents
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_legal_ip_patents_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Legalippatents()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_legal_ip_patents_signal" in res.columns
    assert f"decay_gradient_score_legal_ip_patents_risk_score" in res.columns
    assert not res[f"decay_gradient_score_legal_ip_patents_signal"].isnull().any()

def test_decay_gradient_score_legal_ip_patents_empty():
    extractor = DecayGradientScoreExtractor_Legalippatents()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
