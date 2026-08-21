# Unit Test for EDITransactionFailureCountExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.edi_transaction_failure_count import EDITransactionFailureCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_edi_transaction_failure_count_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EDITransactionFailureCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"edi_transaction_failure_count_signal" in res.columns
    assert f"edi_transaction_failure_count_risk_score" in res.columns
    assert not res[f"edi_transaction_failure_count_signal"].isnull().any()

def test_edi_transaction_failure_count_empty_dataframe():
    extractor = EDITransactionFailureCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
