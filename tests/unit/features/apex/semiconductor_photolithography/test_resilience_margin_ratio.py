# Unit Test for ResilienceMarginRatioExtractor_Semiconductorphotolithography (EUV Semiconductor Photolithography).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.semiconductor_photolithography.resilience_margin_ratio import ResilienceMarginRatioExtractor_Semiconductorphotolithography
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_resilience_margin_ratio_semiconductor_photolithography_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResilienceMarginRatioExtractor_Semiconductorphotolithography()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"resilience_margin_ratio_semiconductor_photolithography_signal" in res.columns
    assert f"resilience_margin_ratio_semiconductor_photolithography_risk_score" in res.columns
    assert not res[f"resilience_margin_ratio_semiconductor_photolithography_signal"].isnull().any()

def test_resilience_margin_ratio_semiconductor_photolithography_empty():
    extractor = ResilienceMarginRatioExtractor_Semiconductorphotolithography()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
