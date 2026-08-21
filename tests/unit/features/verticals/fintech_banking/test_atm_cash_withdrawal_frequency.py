# Unit Test for ATMCashWithdrawalFrequencyExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.atm_cash_withdrawal_frequency import ATMCashWithdrawalFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_atm_cash_withdrawal_frequency_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ATMCashWithdrawalFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"atm_cash_withdrawal_frequency_signal" in res.columns
    assert f"atm_cash_withdrawal_frequency_risk_score" in res.columns
    assert not res[f"atm_cash_withdrawal_frequency_signal"].isnull().any()

def test_atm_cash_withdrawal_frequency_empty_dataframe():
    extractor = ATMCashWithdrawalFrequencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
