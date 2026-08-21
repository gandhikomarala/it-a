# Unit Test for PhysicianPortalActiveMinutesExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.physician_portal_active_minutes import PhysicianPortalActiveMinutesExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_physician_portal_active_minutes_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PhysicianPortalActiveMinutesExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"physician_portal_active_minutes_signal" in res.columns
    assert f"physician_portal_active_minutes_risk_score" in res.columns
    assert not res[f"physician_portal_active_minutes_signal"].isnull().any()

def test_physician_portal_active_minutes_empty_dataframe():
    extractor = PhysicianPortalActiveMinutesExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
