# Comprehensive Unit Test for AEReportingLatencyExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.adverse_event_reporting_latency import AEReportingLatencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_adverse_event_reporting_latency_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AEReportingLatencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"adverse_event_reporting_latency_signal" in res.columns
    assert f"adverse_event_reporting_latency_risk_score" in res.columns
    assert not res[f"adverse_event_reporting_latency_signal"].isnull().any()

def test_adverse_event_reporting_latency_empty_handling():
    extractor = AEReportingLatencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
