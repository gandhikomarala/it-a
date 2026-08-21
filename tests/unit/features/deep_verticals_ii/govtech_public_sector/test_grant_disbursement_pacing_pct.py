# Comprehensive Unit Test for GrantDisbursementPacingExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.grant_disbursement_pacing_pct import GrantDisbursementPacingExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_grant_disbursement_pacing_pct_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = GrantDisbursementPacingExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"grant_disbursement_pacing_pct_signal" in res.columns
    assert f"grant_disbursement_pacing_pct_risk_score" in res.columns
    assert not res[f"grant_disbursement_pacing_pct_signal"].isnull().any()

def test_grant_disbursement_pacing_pct_empty_handling():
    extractor = GrantDisbursementPacingExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
