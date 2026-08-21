# Comprehensive Unit Test for ContractValueLeakagePctExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.contract_value_leakage_pct import ContractValueLeakagePctExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_value_leakage_pct_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractValueLeakagePctExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_value_leakage_pct_signal" in res.columns
    assert f"contract_value_leakage_pct_risk_score" in res.columns
    assert not res[f"contract_value_leakage_pct_signal"].isnull().any()

def test_contract_value_leakage_pct_empty_handling():
    extractor = ContractValueLeakagePctExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
