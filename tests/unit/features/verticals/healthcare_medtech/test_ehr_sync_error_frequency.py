# Unit Test for EHRSyncErrorFrequencyExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.ehr_sync_error_frequency import EHRSyncErrorFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ehr_sync_error_frequency_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EHRSyncErrorFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ehr_sync_error_frequency_signal" in res.columns
    assert f"ehr_sync_error_frequency_risk_score" in res.columns
    assert not res[f"ehr_sync_error_frequency_signal"].isnull().any()

def test_ehr_sync_error_frequency_empty_dataframe():
    extractor = EHRSyncErrorFrequencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
