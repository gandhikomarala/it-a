# Comprehensive Unit Test for BudgetBillingPlanActiveExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.budget_billing_plan_active import BudgetBillingPlanActiveExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_budget_billing_plan_active_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BudgetBillingPlanActiveExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"budget_billing_plan_active_signal" in res.columns
    assert f"budget_billing_plan_active_risk_score" in res.columns
    assert not res[f"budget_billing_plan_active_signal"].isnull().any()

def test_budget_billing_plan_active_empty_handling():
    extractor = BudgetBillingPlanActiveExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
