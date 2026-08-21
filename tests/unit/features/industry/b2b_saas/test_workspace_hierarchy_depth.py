# Unit Test for WorkspaceHierarchyDepth (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.workspace_hierarchy_depth import WorkspaceHierarchyDepth
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_workspace_hierarchy_depth_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = WorkspaceHierarchyDepth()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"workspace_hierarchy_depth_signal" in res.columns
    assert f"workspace_hierarchy_depth_risk_index" in res.columns
    assert not res[f"workspace_hierarchy_depth_signal"].isnull().any()

def test_workspace_hierarchy_depth_empty_handling():
    extractor = WorkspaceHierarchyDepth()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
