# Unit Test for DecayGradientScoreExtractor_Corporatecomplianceethics (Enterprise Ethics Hotline & Whistleblower).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.corporate_compliance_ethics.decay_gradient_score import DecayGradientScoreExtractor_Corporatecomplianceethics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_corporate_compliance_ethics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Corporatecomplianceethics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_corporate_compliance_ethics_signal" in res.columns
    assert f"decay_gradient_score_corporate_compliance_ethics_risk_score" in res.columns
    assert not res[f"decay_gradient_score_corporate_compliance_ethics_signal"].isnull().any()

def test_decay_gradient_score_corporate_compliance_ethics_empty():
    extractor = DecayGradientScoreExtractor_Corporatecomplianceethics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
