# Unit Test for P2PTransferSpikeExtractor (FinTech & Digital Banking).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.fintech_banking.peer_to_peer_transfer_spike import P2PTransferSpikeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_peer_to_peer_transfer_spike_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = P2PTransferSpikeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"peer_to_peer_transfer_spike_signal" in res.columns
    assert f"peer_to_peer_transfer_spike_risk_score" in res.columns
    assert not res[f"peer_to_peer_transfer_spike_signal"].isnull().any()

def test_peer_to_peer_transfer_spike_empty_dataframe():
    extractor = P2PTransferSpikeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
