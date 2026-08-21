# Comprehensive Unit Test for BenefitsEnrollmentFlagExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.benefits_enrollment_completion_flag import BenefitsEnrollmentFlagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_benefits_enrollment_completion_flag_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BenefitsEnrollmentFlagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"benefits_enrollment_completion_flag_signal" in res.columns
    assert f"benefits_enrollment_completion_flag_risk_score" in res.columns
    assert not res[f"benefits_enrollment_completion_flag_signal"].isnull().any()

def test_benefits_enrollment_completion_flag_empty():
    extractor = BenefitsEnrollmentFlagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
