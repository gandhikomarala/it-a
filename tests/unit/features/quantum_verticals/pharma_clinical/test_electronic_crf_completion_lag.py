# Comprehensive Unit Test for ECRFCompletionLagExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.electronic_crf_completion_lag import ECRFCompletionLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electronic_crf_completion_lag_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ECRFCompletionLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electronic_crf_completion_lag_signal" in res.columns
    assert f"electronic_crf_completion_lag_risk_score" in res.columns
    assert not res[f"electronic_crf_completion_lag_signal"].isnull().any()

def test_electronic_crf_completion_lag_empty_handling():
    extractor = ECRFCompletionLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
