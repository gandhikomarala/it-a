# Comprehensive Unit Test for PlanDowngradeAttemptExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.plan_downgrade_attempts import PlanDowngradeAttemptExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_plan_downgrade_attempts_instantiation():
    extractor = PlanDowngradeAttemptExtractor()
    assert extractor.prefix == "plan_downgrade_attempts"

def test_plan_downgrade_attempts_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = PlanDowngradeAttemptExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("plan_downgrade_attempts_")]
    assert len(expected_cols) > 0

def test_plan_downgrade_attempts_transform_empty():
    extractor = PlanDowngradeAttemptExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
