# Unit Test for CIPipelineQueueDurationExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.ci_pipeline_queue_duration import CIPipelineQueueDurationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ci_pipeline_queue_duration_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CIPipelineQueueDurationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ci_pipeline_queue_duration_signal" in res.columns
    assert f"ci_pipeline_queue_duration_risk_score" in res.columns
    assert not res[f"ci_pipeline_queue_duration_signal"].isnull().any()

def test_ci_pipeline_queue_duration_empty_dataframe():
    extractor = CIPipelineQueueDurationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
