# Unit Test for DecayGradientScoreExtractor_Mvnocellulartelecom (MVNO Mobile Virtual Network Operator).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.mvno_cellular_telecom.decay_gradient_score import DecayGradientScoreExtractor_Mvnocellulartelecom
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_mvno_cellular_telecom_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Mvnocellulartelecom()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_mvno_cellular_telecom_signal" in res.columns
    assert f"decay_gradient_score_mvno_cellular_telecom_risk_score" in res.columns
    assert not res[f"decay_gradient_score_mvno_cellular_telecom_signal"].isnull().any()

def test_decay_gradient_score_mvno_cellular_telecom_empty():
    extractor = DecayGradientScoreExtractor_Mvnocellulartelecom()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
