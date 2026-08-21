# Unit Test for CardTransactionFrequencyTrend (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.card_transaction_frequency_trend import CardTransactionFrequencyTrend
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_card_transaction_frequency_trend_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CardTransactionFrequencyTrend()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"card_transaction_frequency_trend_signal" in res.columns
    assert f"card_transaction_frequency_trend_risk_index" in res.columns
    assert not res[f"card_transaction_frequency_trend_signal"].isnull().any()

def test_card_transaction_frequency_trend_empty_handling():
    extractor = CardTransactionFrequencyTrend()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
