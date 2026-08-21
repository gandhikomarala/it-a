# Comprehensive Unit Test for HMMStateTransitionTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.hmm_state_transition import HMMStateTransitionTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_hmm_state_transition_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = HMMStateTransitionTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"hmm_state_transition_transformed" in res.columns
    assert not res[f"hmm_state_transition_transformed"].isnull().any()

def test_hmm_state_transition_numerical_stability():
    transformer = HMMStateTransitionTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"hmm_state_transition_transformed"]).any()
