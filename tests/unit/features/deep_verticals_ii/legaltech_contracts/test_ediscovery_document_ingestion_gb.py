# Comprehensive Unit Test for EDiscoveryIngestionGBExtractor (LegalTech & Contract Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.legaltech_contracts.ediscovery_document_ingestion_gb import EDiscoveryIngestionGBExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ediscovery_document_ingestion_gb_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EDiscoveryIngestionGBExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ediscovery_document_ingestion_gb_signal" in res.columns
    assert f"ediscovery_document_ingestion_gb_risk_score" in res.columns
    assert not res[f"ediscovery_document_ingestion_gb_signal"].isnull().any()

def test_ediscovery_document_ingestion_gb_empty_handling():
    extractor = EDiscoveryIngestionGBExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
