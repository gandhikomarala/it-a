# Unit Test for LoanRepaymentDelinquencyDaysExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.loan_repayment_delinquency_days import LoanRepaymentDelinquencyDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_loan_repayment_delinquency_days_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LoanRepaymentDelinquencyDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"loan_repayment_delinquency_days_signal" in res.columns
    assert f"loan_repayment_delinquency_days_risk_score" in res.columns
    assert not res[f"loan_repayment_delinquency_days_signal"].isnull().any()

def test_loan_repayment_delinquency_days_empty_dataframe():
    extractor = LoanRepaymentDelinquencyDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
