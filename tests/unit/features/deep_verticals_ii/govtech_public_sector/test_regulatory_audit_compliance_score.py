# Comprehensive Unit Test for RegulatoryAuditScoreExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.regulatory_audit_compliance_score import RegulatoryAuditScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_regulatory_audit_compliance_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RegulatoryAuditScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"regulatory_audit_compliance_score_signal" in res.columns
    assert f"regulatory_audit_compliance_score_risk_score" in res.columns
    assert not res[f"regulatory_audit_compliance_score_signal"].isnull().any()

def test_regulatory_audit_compliance_score_empty_handling():
    extractor = RegulatoryAuditScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
