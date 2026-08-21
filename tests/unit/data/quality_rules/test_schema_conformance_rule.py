# Unit Test for SchemaConformanceValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.schema_conformance_rule import SchemaConformanceValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_schema_conformance_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = SchemaConformanceValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
