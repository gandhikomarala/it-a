# Unit Test for SIEMIngestionBufferDelayExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.siem_ingestion_buffer_delay import SIEMIngestionBufferDelayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_siem_ingestion_buffer_delay_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SIEMIngestionBufferDelayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"siem_ingestion_buffer_delay_signal" in res.columns
    assert f"siem_ingestion_buffer_delay_risk_score" in res.columns
    assert not res[f"siem_ingestion_buffer_delay_signal"].isnull().any()

def test_siem_ingestion_buffer_delay_empty_dataframe():
    extractor = SIEMIngestionBufferDelayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
