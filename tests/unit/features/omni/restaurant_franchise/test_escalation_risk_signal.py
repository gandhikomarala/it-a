# Unit Test for EscalationRiskSignalExtractor_Restaurantfranchise (QSR Franchise Store Operations).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.restaurant_franchise.escalation_risk_signal import EscalationRiskSignalExtractor_Restaurantfranchise
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_restaurant_franchise_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Restaurantfranchise()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_restaurant_franchise_signal" in res.columns
    assert f"escalation_risk_signal_restaurant_franchise_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_restaurant_franchise_signal"].isnull().any()

def test_escalation_risk_signal_restaurant_franchise_empty():
    extractor = EscalationRiskSignalExtractor_Restaurantfranchise()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
