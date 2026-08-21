# Comprehensive Unit Test for HandoverFailureHighMobilityExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.handover_failure_rate_high_mobility import HandoverFailureHighMobilityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_handover_failure_rate_high_mobility_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HandoverFailureHighMobilityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"handover_failure_rate_high_mobility_signal" in res.columns
    assert f"handover_failure_rate_high_mobility_risk_score" in res.columns
    assert not res[f"handover_failure_rate_high_mobility_signal"].isnull().any()

def test_handover_failure_rate_high_mobility_empty_handling():
    extractor = HandoverFailureHighMobilityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
