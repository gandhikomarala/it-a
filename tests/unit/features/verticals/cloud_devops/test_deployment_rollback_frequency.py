# Unit Test for DeploymentRollbackFrequencyExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.deployment_rollback_frequency import DeploymentRollbackFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_deployment_rollback_frequency_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DeploymentRollbackFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"deployment_rollback_frequency_signal" in res.columns
    assert f"deployment_rollback_frequency_risk_score" in res.columns
    assert not res[f"deployment_rollback_frequency_signal"].isnull().any()

def test_deployment_rollback_frequency_empty_dataframe():
    extractor = DeploymentRollbackFrequencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
