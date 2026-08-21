# Comprehensive Unit Test for ToxicityReportFrequencyExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.toxicity_report_frequency import ToxicityReportFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_toxicity_report_frequency_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ToxicityReportFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"toxicity_report_frequency_signal" in res.columns
    assert f"toxicity_report_frequency_risk_score" in res.columns
    assert not res[f"toxicity_report_frequency_signal"].isnull().any()

def test_toxicity_report_frequency_empty():
    extractor = ToxicityReportFrequencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
