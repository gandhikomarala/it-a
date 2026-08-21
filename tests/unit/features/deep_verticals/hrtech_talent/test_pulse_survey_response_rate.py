# Comprehensive Unit Test for PulseSurveyResponseRateExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.pulse_survey_response_rate import PulseSurveyResponseRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_pulse_survey_response_rate_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PulseSurveyResponseRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"pulse_survey_response_rate_signal" in res.columns
    assert f"pulse_survey_response_rate_risk_score" in res.columns
    assert not res[f"pulse_survey_response_rate_signal"].isnull().any()

def test_pulse_survey_response_rate_empty():
    extractor = PulseSurveyResponseRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
