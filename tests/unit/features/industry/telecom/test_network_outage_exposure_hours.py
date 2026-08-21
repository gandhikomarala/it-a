# Unit Test for NetworkOutageExposureHours (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.network_outage_exposure_hours import NetworkOutageExposureHours
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_network_outage_exposure_hours_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = NetworkOutageExposureHours()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"network_outage_exposure_hours_signal" in res.columns
    assert f"network_outage_exposure_hours_risk_index" in res.columns
    assert not res[f"network_outage_exposure_hours_signal"].isnull().any()

def test_network_outage_exposure_hours_empty_handling():
    extractor = NetworkOutageExposureHours()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
