# Unit Test for FamilyPlanLineCount (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.family_plan_line_count import FamilyPlanLineCount
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_family_plan_line_count_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FamilyPlanLineCount()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"family_plan_line_count_signal" in res.columns
    assert f"family_plan_line_count_risk_index" in res.columns
    assert not res[f"family_plan_line_count_signal"].isnull().any()

def test_family_plan_line_count_empty_handling():
    extractor = FamilyPlanLineCount()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
