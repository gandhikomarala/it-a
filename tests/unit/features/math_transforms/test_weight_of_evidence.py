# Comprehensive Unit Test for WeightOfEvidenceTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.weight_of_evidence import WeightOfEvidenceTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_weight_of_evidence_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = WeightOfEvidenceTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"weight_of_evidence_transformed" in res.columns
    assert not res[f"weight_of_evidence_transformed"].isnull().any()

def test_weight_of_evidence_numerical_stability():
    transformer = WeightOfEvidenceTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"weight_of_evidence_transformed"]).any()
