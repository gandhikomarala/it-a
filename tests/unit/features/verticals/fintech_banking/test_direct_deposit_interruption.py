# Unit Test for DirectDepositInterruptionExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.direct_deposit_interruption import DirectDepositInterruptionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_direct_deposit_interruption_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DirectDepositInterruptionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"direct_deposit_interruption_signal" in res.columns
    assert f"direct_deposit_interruption_risk_score" in res.columns
    assert not res[f"direct_deposit_interruption_signal"].isnull().any()

def test_direct_deposit_interruption_empty_dataframe():
    extractor = DirectDepositInterruptionExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
