# Unit Test for ShippingDelayExposureCount (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.shipping_delay_exposure_count import ShippingDelayExposureCount
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_shipping_delay_exposure_count_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ShippingDelayExposureCount()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"shipping_delay_exposure_count_signal" in res.columns
    assert f"shipping_delay_exposure_count_risk_index" in res.columns
    assert not res[f"shipping_delay_exposure_count_signal"].isnull().any()

def test_shipping_delay_exposure_count_empty_handling():
    extractor = ShippingDelayExposureCount()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
