# Comprehensive Unit Test for SubscriberRoamingTrafficSplitExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.subscriber_roaming_traffic_split import SubscriberRoamingTrafficSplitExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_subscriber_roaming_traffic_split_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SubscriberRoamingTrafficSplitExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"subscriber_roaming_traffic_split_signal" in res.columns
    assert f"subscriber_roaming_traffic_split_risk_score" in res.columns
    assert not res[f"subscriber_roaming_traffic_split_signal"].isnull().any()

def test_subscriber_roaming_traffic_split_empty_handling():
    extractor = SubscriberRoamingTrafficSplitExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
