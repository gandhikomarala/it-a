# Comprehensive Unit Test for QualityScrapDefectRateExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.quality_scrap_defect_rate import QualityScrapDefectRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quality_scrap_defect_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = QualityScrapDefectRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quality_scrap_defect_rate_signal" in res.columns
    assert f"quality_scrap_defect_rate_risk_score" in res.columns
    assert not res[f"quality_scrap_defect_rate_signal"].isnull().any()

def test_quality_scrap_defect_rate_empty_handling():
    extractor = QualityScrapDefectRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
