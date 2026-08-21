# Comprehensive Unit Test for LeaveOneOutEncoderTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.leave_one_out_encoder import LeaveOneOutEncoderTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_leave_one_out_encoder_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = LeaveOneOutEncoderTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"leave_one_out_encoder_transformed" in res.columns
    assert not res[f"leave_one_out_encoder_transformed"].isnull().any()

def test_leave_one_out_encoder_numerical_stability():
    transformer = LeaveOneOutEncoderTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"leave_one_out_encoder_transformed"]).any()
