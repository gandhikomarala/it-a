# Comprehensive Unit Test for NDAComplianceAuditScoreExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.nda_compliance_audit_score import NDAComplianceAuditScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_nda_compliance_audit_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = NDAComplianceAuditScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"nda_compliance_audit_score_signal" in res.columns
    assert f"nda_compliance_audit_score_risk_score" in res.columns
    assert not res[f"nda_compliance_audit_score_signal"].isnull().any()

def test_nda_compliance_audit_score_empty_handling():
    extractor = NDAComplianceAuditScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
