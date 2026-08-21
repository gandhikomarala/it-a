# Unit Test for PrescriptionRefillOrderLagExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.prescription_refill_order_lag import PrescriptionRefillOrderLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_prescription_refill_order_lag_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PrescriptionRefillOrderLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"prescription_refill_order_lag_signal" in res.columns
    assert f"prescription_refill_order_lag_risk_score" in res.columns
    assert not res[f"prescription_refill_order_lag_signal"].isnull().any()

def test_prescription_refill_order_lag_empty_dataframe():
    extractor = PrescriptionRefillOrderLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
