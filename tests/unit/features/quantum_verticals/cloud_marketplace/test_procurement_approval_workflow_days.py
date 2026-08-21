# Comprehensive Unit Test for ProcurementWorkflowDaysExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.procurement_approval_workflow_days import ProcurementWorkflowDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_procurement_approval_workflow_days_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ProcurementWorkflowDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"procurement_approval_workflow_days_signal" in res.columns
    assert f"procurement_approval_workflow_days_risk_score" in res.columns
    assert not res[f"procurement_approval_workflow_days_signal"].isnull().any()

def test_procurement_approval_workflow_days_empty_handling():
    extractor = ProcurementWorkflowDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
