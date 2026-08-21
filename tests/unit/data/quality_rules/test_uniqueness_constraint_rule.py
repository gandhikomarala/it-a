# Unit Test for UniquenessConstraintValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.uniqueness_constraint_rule import UniquenessConstraintValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_uniqueness_constraint_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = UniquenessConstraintValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
