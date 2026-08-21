# Comprehensive Unit Test for OrgReorganizationExposureExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.org_chart_reorganization_exposure import OrgReorganizationExposureExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_org_chart_reorganization_exposure_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OrgReorganizationExposureExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"org_chart_reorganization_exposure_signal" in res.columns
    assert f"org_chart_reorganization_exposure_risk_score" in res.columns
    assert not res[f"org_chart_reorganization_exposure_signal"].isnull().any()

def test_org_chart_reorganization_exposure_empty():
    extractor = OrgReorganizationExposureExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
