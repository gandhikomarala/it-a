# Comprehensive Unit Test for ProtocolAmendmentImpactExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.protocol_amendment_impact_score import ProtocolAmendmentImpactExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_protocol_amendment_impact_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ProtocolAmendmentImpactExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"protocol_amendment_impact_score_signal" in res.columns
    assert f"protocol_amendment_impact_score_risk_score" in res.columns
    assert not res[f"protocol_amendment_impact_score_signal"].isnull().any()

def test_protocol_amendment_impact_score_empty_handling():
    extractor = ProtocolAmendmentImpactExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
