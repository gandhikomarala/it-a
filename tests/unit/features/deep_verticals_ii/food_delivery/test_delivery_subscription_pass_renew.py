# Comprehensive Unit Test for DeliveryPassRenewExtractor (Food Delivery & Quick-Service).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.food_delivery.delivery_subscription_pass_renew import DeliveryPassRenewExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_delivery_subscription_pass_renew_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DeliveryPassRenewExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"delivery_subscription_pass_renew_signal" in res.columns
    assert f"delivery_subscription_pass_renew_risk_score" in res.columns
    assert not res[f"delivery_subscription_pass_renew_signal"].isnull().any()

def test_delivery_subscription_pass_renew_empty_handling():
    extractor = DeliveryPassRenewExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
