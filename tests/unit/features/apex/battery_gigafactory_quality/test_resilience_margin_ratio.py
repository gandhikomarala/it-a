# Unit Test for ResilienceMarginRatioExtractor_Batterygigafactoryquality (Lithium-Ion Battery Gigafactory Cell QC).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.battery_gigafactory_quality.resilience_margin_ratio import ResilienceMarginRatioExtractor_Batterygigafactoryquality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_resilience_margin_ratio_battery_gigafactory_quality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResilienceMarginRatioExtractor_Batterygigafactoryquality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"resilience_margin_ratio_battery_gigafactory_quality_signal" in res.columns
    assert f"resilience_margin_ratio_battery_gigafactory_quality_risk_score" in res.columns
    assert not res[f"resilience_margin_ratio_battery_gigafactory_quality_signal"].isnull().any()

def test_resilience_margin_ratio_battery_gigafactory_quality_empty():
    extractor = ResilienceMarginRatioExtractor_Batterygigafactoryquality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
