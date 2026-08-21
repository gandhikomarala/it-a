# Unit tests for data loading, profiling, and preprocessing.
import pytest
import numpy as np
import pandas as pd
from ml.data.synthetic_generator import SyntheticCustomerGenerator
from ml.data.profiler import DatasetProfiler
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.imputers import AdaptiveImputer
from ml.preprocessing.scalers import FeatureScaler

def test_synthetic_data_generation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(100)
    assert len(df) == 100
    assert "churn" in df.columns
    assert "customer_id" in df.columns
    assert df["churn"].nunique() == 2

def test_dataset_profiler():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    profiler = DatasetProfiler(df)
    report = profiler.evaluate_quality()
    assert report.quality_score > 70.0
    assert report.is_approved is True

def test_preprocessing_pipeline_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(100)
    X = df.drop(columns=["churn"])
    
    pipeline = PreprocessingPipeline()
    X_trans = pipeline.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert not X_trans.isnull().any().any()
    assert len(X_trans) == 100
