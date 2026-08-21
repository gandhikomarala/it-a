# Unit Test for NullThresholdValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.null_threshold_rule import NullThresholdValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_null_threshold_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = NullThresholdValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
