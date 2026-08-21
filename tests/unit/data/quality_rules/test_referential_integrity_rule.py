# Unit Test for ReferentialIntegrityValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.referential_integrity_rule import ReferentialIntegrityValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_referential_integrity_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = ReferentialIntegrityValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
