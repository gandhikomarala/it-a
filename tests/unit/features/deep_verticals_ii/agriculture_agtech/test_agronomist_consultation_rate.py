# Comprehensive Unit Test for AgronomistConsultationRateExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.agronomist_consultation_rate import AgronomistConsultationRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_agronomist_consultation_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AgronomistConsultationRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"agronomist_consultation_rate_signal" in res.columns
    assert f"agronomist_consultation_rate_risk_score" in res.columns
    assert not res[f"agronomist_consultation_rate_signal"].isnull().any()

def test_agronomist_consultation_rate_empty_handling():
    extractor = AgronomistConsultationRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
