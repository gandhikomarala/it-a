# Comprehensive Unit Test for PrivateOfferDiscountDepthExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.private_offer_discount_depth import PrivateOfferDiscountDepthExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_private_offer_discount_depth_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PrivateOfferDiscountDepthExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"private_offer_discount_depth_signal" in res.columns
    assert f"private_offer_discount_depth_risk_score" in res.columns
    assert not res[f"private_offer_discount_depth_signal"].isnull().any()

def test_private_offer_discount_depth_empty_handling():
    extractor = PrivateOfferDiscountDepthExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
