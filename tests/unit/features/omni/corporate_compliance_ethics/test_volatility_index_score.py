# Unit Test for VolatilityIndexScoreExtractor_Corporatecomplianceethics (Enterprise Ethics Hotline & Whistleblower).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.corporate_compliance_ethics.volatility_index_score import VolatilityIndexScoreExtractor_Corporatecomplianceethics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_corporate_compliance_ethics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Corporatecomplianceethics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_corporate_compliance_ethics_signal" in res.columns
    assert f"volatility_index_score_corporate_compliance_ethics_risk_score" in res.columns
    assert not res[f"volatility_index_score_corporate_compliance_ethics_signal"].isnull().any()

def test_volatility_index_score_corporate_compliance_ethics_empty():
    extractor = VolatilityIndexScoreExtractor_Corporatecomplianceethics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
