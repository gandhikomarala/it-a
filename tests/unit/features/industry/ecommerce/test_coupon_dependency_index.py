# Unit Test for CouponDependencyIndex (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.coupon_dependency_index import CouponDependencyIndex
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_coupon_dependency_index_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CouponDependencyIndex()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"coupon_dependency_index_signal" in res.columns
    assert f"coupon_dependency_index_risk_index" in res.columns
    assert not res[f"coupon_dependency_index_signal"].isnull().any()

def test_coupon_dependency_index_empty_handling():
    extractor = CouponDependencyIndex()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
