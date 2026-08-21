# Unit Test for LifecycleBurnRateExtractor_Hyperscaledatacenters (Hyperscale Datacenter Liquid Cooling).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hyperscale_datacenters.lifecycle_burn_rate import LifecycleBurnRateExtractor_Hyperscaledatacenters
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_hyperscale_datacenters_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Hyperscaledatacenters()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_hyperscale_datacenters_signal" in res.columns
    assert f"lifecycle_burn_rate_hyperscale_datacenters_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_hyperscale_datacenters_signal"].isnull().any()

def test_lifecycle_burn_rate_hyperscale_datacenters_empty():
    extractor = LifecycleBurnRateExtractor_Hyperscaledatacenters()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
