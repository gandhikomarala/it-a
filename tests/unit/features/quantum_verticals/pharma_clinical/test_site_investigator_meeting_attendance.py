# Comprehensive Unit Test for SiteMeetingAttendanceExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.site_investigator_meeting_attendance import SiteMeetingAttendanceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_site_investigator_meeting_attendance_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SiteMeetingAttendanceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"site_investigator_meeting_attendance_signal" in res.columns
    assert f"site_investigator_meeting_attendance_risk_score" in res.columns
    assert not res[f"site_investigator_meeting_attendance_signal"].isnull().any()

def test_site_investigator_meeting_attendance_empty_handling():
    extractor = SiteMeetingAttendanceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
