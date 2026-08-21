# Unit Test for WireTransferOutflowRatio (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.wire_transfer_outflow_ratio import WireTransferOutflowRatio
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_wire_transfer_outflow_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = WireTransferOutflowRatio()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"wire_transfer_outflow_ratio_signal" in res.columns
    assert f"wire_transfer_outflow_ratio_risk_index" in res.columns
    assert not res[f"wire_transfer_outflow_ratio_signal"].isnull().any()

def test_wire_transfer_outflow_ratio_empty_handling():
    extractor = WireTransferOutflowRatio()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
