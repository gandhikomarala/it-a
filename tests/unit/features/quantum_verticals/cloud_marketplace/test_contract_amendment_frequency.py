# Comprehensive Unit Test for ContractAmendmentFrequencyExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.contract_amendment_frequency import ContractAmendmentFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_amendment_frequency_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractAmendmentFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_amendment_frequency_signal" in res.columns
    assert f"contract_amendment_frequency_risk_score" in res.columns
    assert not res[f"contract_amendment_frequency_signal"].isnull().any()

def test_contract_amendment_frequency_empty_handling():
    extractor = ContractAmendmentFrequencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
