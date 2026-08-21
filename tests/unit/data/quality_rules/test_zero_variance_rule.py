# Unit Test for ZeroVarianceValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.zero_variance_rule import ZeroVarianceValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_zero_variance_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = ZeroVarianceValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
