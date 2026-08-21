# Unit Test for DepositOutflowAccelerationExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.deposit_outflow_acceleration import DepositOutflowAccelerationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_deposit_outflow_acceleration_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DepositOutflowAccelerationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"deposit_outflow_acceleration_signal" in res.columns
    assert f"deposit_outflow_acceleration_risk_score" in res.columns
    assert not res[f"deposit_outflow_acceleration_signal"].isnull().any()

def test_deposit_outflow_acceleration_empty_dataframe():
    extractor = DepositOutflowAccelerationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
