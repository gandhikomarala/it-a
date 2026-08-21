# Unit Test for CartAbandonmentVelocity (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.cart_abandonment_velocity import CartAbandonmentVelocity
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cart_abandonment_velocity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CartAbandonmentVelocity()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cart_abandonment_velocity_signal" in res.columns
    assert f"cart_abandonment_velocity_risk_index" in res.columns
    assert not res[f"cart_abandonment_velocity_signal"].isnull().any()

def test_cart_abandonment_velocity_empty_handling():
    extractor = CartAbandonmentVelocity()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
