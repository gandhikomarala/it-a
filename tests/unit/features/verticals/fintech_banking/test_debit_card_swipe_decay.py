# Unit Test for DebitCardSwipeDecayExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.debit_card_swipe_decay import DebitCardSwipeDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_debit_card_swipe_decay_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DebitCardSwipeDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"debit_card_swipe_decay_signal" in res.columns
    assert f"debit_card_swipe_decay_risk_score" in res.columns
    assert not res[f"debit_card_swipe_decay_signal"].isnull().any()

def test_debit_card_swipe_decay_empty_dataframe():
    extractor = DebitCardSwipeDecayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
