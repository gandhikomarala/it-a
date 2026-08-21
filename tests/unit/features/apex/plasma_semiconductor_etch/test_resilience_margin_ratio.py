# Unit Test for ResilienceMarginRatioExtractor_Plasmasemiconductoretch (Atomic Layer Semiconductor Plasma Etch).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.plasma_semiconductor_etch.resilience_margin_ratio import ResilienceMarginRatioExtractor_Plasmasemiconductoretch
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_resilience_margin_ratio_plasma_semiconductor_etch_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResilienceMarginRatioExtractor_Plasmasemiconductoretch()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"resilience_margin_ratio_plasma_semiconductor_etch_signal" in res.columns
    assert f"resilience_margin_ratio_plasma_semiconductor_etch_risk_score" in res.columns
    assert not res[f"resilience_margin_ratio_plasma_semiconductor_etch_signal"].isnull().any()

def test_resilience_margin_ratio_plasma_semiconductor_etch_empty():
    extractor = ResilienceMarginRatioExtractor_Plasmasemiconductoretch()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
