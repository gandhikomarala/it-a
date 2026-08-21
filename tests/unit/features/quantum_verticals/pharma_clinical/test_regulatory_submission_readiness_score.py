# Comprehensive Unit Test for RegulatoryReadinessScoreExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.regulatory_submission_readiness_score import RegulatoryReadinessScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_regulatory_submission_readiness_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RegulatoryReadinessScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"regulatory_submission_readiness_score_signal" in res.columns
    assert f"regulatory_submission_readiness_score_risk_score" in res.columns
    assert not res[f"regulatory_submission_readiness_score_signal"].isnull().any()

def test_regulatory_submission_readiness_score_empty_handling():
    extractor = RegulatoryReadinessScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
