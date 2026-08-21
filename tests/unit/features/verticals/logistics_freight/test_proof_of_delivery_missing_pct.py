# Unit Test for ProofOfDeliveryMissingPctExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.proof_of_delivery_missing_pct import ProofOfDeliveryMissingPctExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_proof_of_delivery_missing_pct_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ProofOfDeliveryMissingPctExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"proof_of_delivery_missing_pct_signal" in res.columns
    assert f"proof_of_delivery_missing_pct_risk_score" in res.columns
    assert not res[f"proof_of_delivery_missing_pct_signal"].isnull().any()

def test_proof_of_delivery_missing_pct_empty_dataframe():
    extractor = ProofOfDeliveryMissingPctExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
