# Unit Test for AuditLogExportRate (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.audit_log_export_rate import AuditLogExportRate
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_audit_log_export_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AuditLogExportRate()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"audit_log_export_rate_signal" in res.columns
    assert f"audit_log_export_rate_risk_index" in res.columns
    assert not res[f"audit_log_export_rate_signal"].isnull().any()

def test_audit_log_export_rate_empty_handling():
    extractor = AuditLogExportRate()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
