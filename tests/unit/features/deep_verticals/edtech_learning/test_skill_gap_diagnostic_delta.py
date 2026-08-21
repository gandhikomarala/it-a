# Comprehensive Unit Test for SkillGapDiagnosticDeltaExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.skill_gap_diagnostic_delta import SkillGapDiagnosticDeltaExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_skill_gap_diagnostic_delta_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SkillGapDiagnosticDeltaExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"skill_gap_diagnostic_delta_signal" in res.columns
    assert f"skill_gap_diagnostic_delta_risk_score" in res.columns
    assert not res[f"skill_gap_diagnostic_delta_signal"].isnull().any()

def test_skill_gap_diagnostic_delta_empty():
    extractor = SkillGapDiagnosticDeltaExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
