# Unit Test for RangeBoundaryValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.range_boundary_rule import RangeBoundaryValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_range_boundary_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = RangeBoundaryValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
