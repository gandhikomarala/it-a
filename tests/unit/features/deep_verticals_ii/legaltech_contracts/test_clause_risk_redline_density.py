# Comprehensive Unit Test for ClauseRiskRedlineDensityExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.clause_risk_redline_density import ClauseRiskRedlineDensityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_clause_risk_redline_density_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ClauseRiskRedlineDensityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"clause_risk_redline_density_signal" in res.columns
    assert f"clause_risk_redline_density_risk_score" in res.columns
    assert not res[f"clause_risk_redline_density_signal"].isnull().any()

def test_clause_risk_redline_density_empty_handling():
    extractor = ClauseRiskRedlineDensityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
