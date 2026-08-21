# Unit Test for ResilienceMarginRatioExtractor_Vppmicrogrids (Virtual Power Plants & Microgrid Orchestration).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.vpp_microgrids.resilience_margin_ratio import ResilienceMarginRatioExtractor_Vppmicrogrids
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_resilience_margin_ratio_vpp_microgrids_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResilienceMarginRatioExtractor_Vppmicrogrids()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"resilience_margin_ratio_vpp_microgrids_signal" in res.columns
    assert f"resilience_margin_ratio_vpp_microgrids_risk_score" in res.columns
    assert not res[f"resilience_margin_ratio_vpp_microgrids_signal"].isnull().any()

def test_resilience_margin_ratio_vpp_microgrids_empty():
    extractor = ResilienceMarginRatioExtractor_Vppmicrogrids()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
