# Unit Test for IAMPermissionSprawlScoreExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.iam_permission_sprawl_score import IAMPermissionSprawlScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_iam_permission_sprawl_score_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = IAMPermissionSprawlScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"iam_permission_sprawl_score_signal" in res.columns
    assert f"iam_permission_sprawl_score_risk_score" in res.columns
    assert not res[f"iam_permission_sprawl_score_signal"].isnull().any()

def test_iam_permission_sprawl_score_empty_dataframe():
    extractor = IAMPermissionSprawlScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
