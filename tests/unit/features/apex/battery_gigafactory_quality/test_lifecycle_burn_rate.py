# Unit Test for LifecycleBurnRateExtractor_Batterygigafactoryquality (Lithium-Ion Battery Gigafactory Cell QC).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.battery_gigafactory_quality.lifecycle_burn_rate import LifecycleBurnRateExtractor_Batterygigafactoryquality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_battery_gigafactory_quality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Batterygigafactoryquality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_battery_gigafactory_quality_signal" in res.columns
    assert f"lifecycle_burn_rate_battery_gigafactory_quality_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_battery_gigafactory_quality_signal"].isnull().any()

def test_lifecycle_burn_rate_battery_gigafactory_quality_empty():
    extractor = LifecycleBurnRateExtractor_Batterygigafactoryquality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
