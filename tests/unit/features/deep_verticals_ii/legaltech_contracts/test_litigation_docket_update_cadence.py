# Comprehensive Unit Test for LitigationDocketUpdateCadenceExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.litigation_docket_update_cadence import LitigationDocketUpdateCadenceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_litigation_docket_update_cadence_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LitigationDocketUpdateCadenceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"litigation_docket_update_cadence_signal" in res.columns
    assert f"litigation_docket_update_cadence_risk_score" in res.columns
    assert not res[f"litigation_docket_update_cadence_signal"].isnull().any()

def test_litigation_docket_update_cadence_empty_handling():
    extractor = LitigationDocketUpdateCadenceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
