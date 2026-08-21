# Unit Test for TemporalMonotonicityValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.temporal_monotonicity_rule import TemporalMonotonicityValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_temporal_monotonicity_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = TemporalMonotonicityValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
