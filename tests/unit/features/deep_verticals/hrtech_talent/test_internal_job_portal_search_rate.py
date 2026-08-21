# Comprehensive Unit Test for InternalJobSearchRateExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.internal_job_portal_search_rate import InternalJobSearchRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_internal_job_portal_search_rate_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = InternalJobSearchRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"internal_job_portal_search_rate_signal" in res.columns
    assert f"internal_job_portal_search_rate_risk_score" in res.columns
    assert not res[f"internal_job_portal_search_rate_signal"].isnull().any()

def test_internal_job_portal_search_rate_empty():
    extractor = InternalJobSearchRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
