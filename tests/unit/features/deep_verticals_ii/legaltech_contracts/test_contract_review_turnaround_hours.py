# Comprehensive Unit Test for ContractReviewTurnaroundHoursExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.contract_review_turnaround_hours import ContractReviewTurnaroundHoursExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_review_turnaround_hours_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractReviewTurnaroundHoursExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_review_turnaround_hours_signal" in res.columns
    assert f"contract_review_turnaround_hours_risk_score" in res.columns
    assert not res[f"contract_review_turnaround_hours_signal"].isnull().any()

def test_contract_review_turnaround_hours_empty_handling():
    extractor = ContractReviewTurnaroundHoursExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
