# Unit Test for AppointmentNoShowRateExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.appointment_no_show_rate import AppointmentNoShowRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_appointment_no_show_rate_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AppointmentNoShowRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"appointment_no_show_rate_signal" in res.columns
    assert f"appointment_no_show_rate_risk_score" in res.columns
    assert not res[f"appointment_no_show_rate_signal"].isnull().any()

def test_appointment_no_show_rate_empty_dataframe():
    extractor = AppointmentNoShowRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
