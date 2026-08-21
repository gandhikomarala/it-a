# Comprehensive Unit Test for Manager1on1CadenceExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.manager_1on1_meeting_cadence import Manager1on1CadenceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_manager_1on1_meeting_cadence_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = Manager1on1CadenceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"manager_1on1_meeting_cadence_signal" in res.columns
    assert f"manager_1on1_meeting_cadence_risk_score" in res.columns
    assert not res[f"manager_1on1_meeting_cadence_signal"].isnull().any()

def test_manager_1on1_meeting_cadence_empty():
    extractor = Manager1on1CadenceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
