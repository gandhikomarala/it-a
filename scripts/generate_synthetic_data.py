#!/usr/bin/env python3
# CLI runner for generating synthetic customer datasets.
import os
import argparse
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customer dataset for churn modeling.")
    parser.add_argument("--customers", type=int, default=5000, help="Number of customer records to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_customers.csv", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    gen = SyntheticCustomerGenerator(random_seed=args.seed)
    df = gen.generate(n_customers=args.customers)
    
    if args.output.endswith(".parquet"):
        df.to_parquet(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"Successfully generated {len(df):,} customer records -> {args.output}")

if __name__ == '__main__':
    main()
