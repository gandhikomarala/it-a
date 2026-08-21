# Unit Test for WishlistConversionRatio (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.wishlist_to_cart_conversion_ratio import WishlistConversionRatio
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_wishlist_to_cart_conversion_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = WishlistConversionRatio()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"wishlist_to_cart_conversion_ratio_signal" in res.columns
    assert f"wishlist_to_cart_conversion_ratio_risk_index" in res.columns
    assert not res[f"wishlist_to_cart_conversion_ratio_signal"].isnull().any()

def test_wishlist_to_cart_conversion_ratio_empty_handling():
    extractor = WishlistConversionRatio()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
