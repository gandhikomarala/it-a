# Comprehensive Unit Test for EDCQueryResolutionTurnaroundExtractor (Pharma & Clinical Trial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.pharma_clinical.edc_query_resolution_turnaround import EDCQueryResolutionTurnaroundExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_edc_query_resolution_turnaround_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EDCQueryResolutionTurnaroundExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"edc_query_resolution_turnaround_signal" in res.columns
    assert f"edc_query_resolution_turnaround_risk_score" in res.columns
    assert not res[f"edc_query_resolution_turnaround_signal"].isnull().any()

def test_edc_query_resolution_turnaround_empty_handling():
    extractor = EDCQueryResolutionTurnaroundExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
