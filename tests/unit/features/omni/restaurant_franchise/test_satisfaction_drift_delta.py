# Unit Test for SatisfactionDriftDeltaExtractor_Restaurantfranchise (QSR Franchise Store Operations).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.restaurant_franchise.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Restaurantfranchise
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_restaurant_franchise_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Restaurantfranchise()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_restaurant_franchise_signal" in res.columns
    assert f"satisfaction_drift_delta_restaurant_franchise_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_restaurant_franchise_signal"].isnull().any()

def test_satisfaction_drift_delta_restaurant_franchise_empty():
    extractor = SatisfactionDriftDeltaExtractor_Restaurantfranchise()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
