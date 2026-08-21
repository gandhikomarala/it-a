# Unit Test for ReturnMerchandiseRate (ecommerce).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.ecommerce.return_merchandise_rate import ReturnMerchandiseRate
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_return_merchandise_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ReturnMerchandiseRate()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"return_merchandise_rate_signal" in res.columns
    assert f"return_merchandise_rate_risk_index" in res.columns
    assert not res[f"return_merchandise_rate_signal"].isnull().any()

def test_return_merchandise_rate_empty_handling():
    extractor = ReturnMerchandiseRate()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
