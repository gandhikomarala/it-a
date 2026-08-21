# Comprehensive Unit Test for MultiPolicyBundlingRatioExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.multi_policy_bundling_ratio import MultiPolicyBundlingRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_multi_policy_bundling_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MultiPolicyBundlingRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"multi_policy_bundling_ratio_signal" in res.columns
    assert f"multi_policy_bundling_ratio_risk_score" in res.columns
    assert not res[f"multi_policy_bundling_ratio_signal"].isnull().any()

def test_multi_policy_bundling_ratio_empty_handling():
    extractor = MultiPolicyBundlingRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
