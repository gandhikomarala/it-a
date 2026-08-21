# Comprehensive Unit Test for MultiCloudBillingSplitExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.multi_cloud_billing_split_ratio import MultiCloudBillingSplitExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_multi_cloud_billing_split_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MultiCloudBillingSplitExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"multi_cloud_billing_split_ratio_signal" in res.columns
    assert f"multi_cloud_billing_split_ratio_risk_score" in res.columns
    assert not res[f"multi_cloud_billing_split_ratio_signal"].isnull().any()

def test_multi_cloud_billing_split_ratio_empty_handling():
    extractor = MultiCloudBillingSplitExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
