# Unit Test for PatientIntakeFormDropoffExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.patient_intake_form_dropoff import PatientIntakeFormDropoffExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_patient_intake_form_dropoff_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PatientIntakeFormDropoffExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"patient_intake_form_dropoff_signal" in res.columns
    assert f"patient_intake_form_dropoff_risk_score" in res.columns
    assert not res[f"patient_intake_form_dropoff_signal"].isnull().any()

def test_patient_intake_form_dropoff_empty_dataframe():
    extractor = PatientIntakeFormDropoffExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
