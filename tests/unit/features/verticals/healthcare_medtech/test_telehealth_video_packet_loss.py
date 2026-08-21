# Unit Test for TelehealthVideoPacketLossExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.telehealth_video_packet_loss import TelehealthVideoPacketLossExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telehealth_video_packet_loss_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelehealthVideoPacketLossExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telehealth_video_packet_loss_signal" in res.columns
    assert f"telehealth_video_packet_loss_risk_score" in res.columns
    assert not res[f"telehealth_video_packet_loss_signal"].isnull().any()

def test_telehealth_video_packet_loss_empty_dataframe():
    extractor = TelehealthVideoPacketLossExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
