# Unit Test for ResilienceMarginRatioExtractor_Smartgridsynchrophasor (Smart Grid PMU Synchrophasor Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_grid_synchrophasor.resilience_margin_ratio import ResilienceMarginRatioExtractor_Smartgridsynchrophasor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_resilience_margin_ratio_smart_grid_synchrophasor_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResilienceMarginRatioExtractor_Smartgridsynchrophasor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"resilience_margin_ratio_smart_grid_synchrophasor_signal" in res.columns
    assert f"resilience_margin_ratio_smart_grid_synchrophasor_risk_score" in res.columns
    assert not res[f"resilience_margin_ratio_smart_grid_synchrophasor_signal"].isnull().any()

def test_resilience_margin_ratio_smart_grid_synchrophasor_empty():
    extractor = ResilienceMarginRatioExtractor_Smartgridsynchrophasor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
