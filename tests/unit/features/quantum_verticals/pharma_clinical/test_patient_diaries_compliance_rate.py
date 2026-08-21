# Comprehensive Unit Test for PatientDiariesComplianceExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.patient_diaries_compliance_rate import PatientDiariesComplianceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_patient_diaries_compliance_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PatientDiariesComplianceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"patient_diaries_compliance_rate_signal" in res.columns
    assert f"patient_diaries_compliance_rate_risk_score" in res.columns
    assert not res[f"patient_diaries_compliance_rate_signal"].isnull().any()

def test_patient_diaries_compliance_rate_empty_handling():
    extractor = PatientDiariesComplianceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
