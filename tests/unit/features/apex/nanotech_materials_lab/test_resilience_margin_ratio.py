# Unit Test for ResilienceMarginRatioExtractor_Nanotechmaterialslab (Advanced Nanomaterials Synthesis Lab).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.nanotech_materials_lab.resilience_margin_ratio import ResilienceMarginRatioExtractor_Nanotechmaterialslab
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_resilience_margin_ratio_nanotech_materials_lab_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResilienceMarginRatioExtractor_Nanotechmaterialslab()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"resilience_margin_ratio_nanotech_materials_lab_signal" in res.columns
    assert f"resilience_margin_ratio_nanotech_materials_lab_risk_score" in res.columns
    assert not res[f"resilience_margin_ratio_nanotech_materials_lab_signal"].isnull().any()

def test_resilience_margin_ratio_nanotech_materials_lab_empty():
    extractor = ResilienceMarginRatioExtractor_Nanotechmaterialslab()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
