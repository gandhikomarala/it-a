# Comprehensive Unit Test for TargetMeanEncoderTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.target_mean_encoder import TargetMeanEncoderTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_target_mean_encoder_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = TargetMeanEncoderTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"target_mean_encoder_transformed" in res.columns
    assert not res[f"target_mean_encoder_transformed"].isnull().any()

def test_target_mean_encoder_numerical_stability():
    transformer = TargetMeanEncoderTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"target_mean_encoder_transformed"]).any()
