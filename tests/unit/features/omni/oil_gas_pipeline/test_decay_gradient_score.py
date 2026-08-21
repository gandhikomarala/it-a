# Unit Test for DecayGradientScoreExtractor_Oilgaspipeline (Oil & Gas Pipeline Integrity).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.oil_gas_pipeline.decay_gradient_score import DecayGradientScoreExtractor_Oilgaspipeline
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_oil_gas_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Oilgaspipeline()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_oil_gas_pipeline_signal" in res.columns
    assert f"decay_gradient_score_oil_gas_pipeline_risk_score" in res.columns
    assert not res[f"decay_gradient_score_oil_gas_pipeline_signal"].isnull().any()

def test_decay_gradient_score_oil_gas_pipeline_empty():
    extractor = DecayGradientScoreExtractor_Oilgaspipeline()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
