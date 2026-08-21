# Comprehensive Unit Test for ArbitrationPrevalenceRatioExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.arbitration_clause_prevalence_ratio import ArbitrationPrevalenceRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_arbitration_clause_prevalence_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ArbitrationPrevalenceRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"arbitration_clause_prevalence_ratio_signal" in res.columns
    assert f"arbitration_clause_prevalence_ratio_risk_score" in res.columns
    assert not res[f"arbitration_clause_prevalence_ratio_signal"].isnull().any()

def test_arbitration_clause_prevalence_ratio_empty_handling():
    extractor = ArbitrationPrevalenceRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
