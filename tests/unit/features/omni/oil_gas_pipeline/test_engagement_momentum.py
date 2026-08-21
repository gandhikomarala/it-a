# Unit Test for EngagementMomentumExtractor_Oilgaspipeline (Oil & Gas Pipeline Integrity).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.oil_gas_pipeline.engagement_momentum import EngagementMomentumExtractor_Oilgaspipeline
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_oil_gas_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Oilgaspipeline()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_oil_gas_pipeline_signal" in res.columns
    assert f"engagement_momentum_oil_gas_pipeline_risk_score" in res.columns
    assert not res[f"engagement_momentum_oil_gas_pipeline_signal"].isnull().any()

def test_engagement_momentum_oil_gas_pipeline_empty():
    extractor = EngagementMomentumExtractor_Oilgaspipeline()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
