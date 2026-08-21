# Unit Test for DirectDepositSwitchDetected (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.direct_deposit_switch_detected import DirectDepositSwitchDetected
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_direct_deposit_switch_detected_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DirectDepositSwitchDetected()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"direct_deposit_switch_detected_signal" in res.columns
    assert f"direct_deposit_switch_detected_risk_index" in res.columns
    assert not res[f"direct_deposit_switch_detected_signal"].isnull().any()

def test_direct_deposit_switch_detected_empty_handling():
    extractor = DirectDepositSwitchDetected()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
