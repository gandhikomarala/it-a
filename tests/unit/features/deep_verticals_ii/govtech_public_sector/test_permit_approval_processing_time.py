# Comprehensive Unit Test for PermitProcessingTimeExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.permit_approval_processing_time import PermitProcessingTimeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_permit_approval_processing_time_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PermitProcessingTimeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"permit_approval_processing_time_signal" in res.columns
    assert f"permit_approval_processing_time_risk_score" in res.columns
    assert not res[f"permit_approval_processing_time_signal"].isnull().any()

def test_permit_approval_processing_time_empty_handling():
    extractor = PermitProcessingTimeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
