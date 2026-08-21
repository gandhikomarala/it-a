# Unit Test for CostBudgetOverrunPctExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.cost_budget_overrun_pct import CostBudgetOverrunPctExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cost_budget_overrun_pct_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CostBudgetOverrunPctExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cost_budget_overrun_pct_signal" in res.columns
    assert f"cost_budget_overrun_pct_risk_score" in res.columns
    assert not res[f"cost_budget_overrun_pct_signal"].isnull().any()

def test_cost_budget_overrun_pct_empty_dataframe():
    extractor = CostBudgetOverrunPctExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
