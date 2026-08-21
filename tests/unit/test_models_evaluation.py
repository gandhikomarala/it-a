# Unit tests for ML models and evaluation metrics.
import pytest
import numpy as np
import pandas as pd
from ml.data.synthetic_generator import SyntheticCustomerGenerator
from ml.training.orchestrator import TrainingOrchestrator
from ml.evaluation.evaluator import ModelEvaluator
from ml.monitoring.drift_detector import StatisticalDriftDetector

def test_model_training_and_evaluation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(500)
    pipeline, model, metrics = TrainingOrchestrator.train_and_evaluate(df, algorithm="LightGBM")
    
    assert metrics.roc_auc >= 0.65
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0

def test_drift_detector():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df_base = gen.generate(100)
    df_curr = gen.generate(100)
    
    detector = StatisticalDriftDetector()
    report = detector.calculate_drift(df_base, df_curr)
    assert report.overall_drift_status.value in ["NORMAL", "WARNING", "CRITICAL"]
    assert report.total_features_monitored > 0
