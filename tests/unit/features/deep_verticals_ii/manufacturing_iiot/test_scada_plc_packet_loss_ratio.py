# Comprehensive Unit Test for SCADAPacketLossRatioExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.scada_plc_packet_loss_ratio import SCADAPacketLossRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_scada_plc_packet_loss_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SCADAPacketLossRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"scada_plc_packet_loss_ratio_signal" in res.columns
    assert f"scada_plc_packet_loss_ratio_risk_score" in res.columns
    assert not res[f"scada_plc_packet_loss_ratio_signal"].isnull().any()

def test_scada_plc_packet_loss_ratio_empty_handling():
    extractor = SCADAPacketLossRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
