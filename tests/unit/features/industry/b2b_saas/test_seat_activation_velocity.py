# Unit Test for SeatActivationVelocity (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.seat_activation_velocity import SeatActivationVelocity
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_seat_activation_velocity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SeatActivationVelocity()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"seat_activation_velocity_signal" in res.columns
    assert f"seat_activation_velocity_risk_index" in res.columns
    assert not res[f"seat_activation_velocity_signal"].isnull().any()

def test_seat_activation_velocity_empty_handling():
    extractor = SeatActivationVelocity()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
