# Comprehensive Unit Test for OfficeHoursAttendanceExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.instructor_office_hours_attendance import OfficeHoursAttendanceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_instructor_office_hours_attendance_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OfficeHoursAttendanceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"instructor_office_hours_attendance_signal" in res.columns
    assert f"instructor_office_hours_attendance_risk_score" in res.columns
    assert not res[f"instructor_office_hours_attendance_signal"].isnull().any()

def test_instructor_office_hours_attendance_empty():
    extractor = OfficeHoursAttendanceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
