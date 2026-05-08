import pandas as pd
import argparse

def combine_and_deduplicate(file1: str, file2: str, output: str, key_columns: list = None):
    """
    Combine two parquet files and remove duplicates
    
    Args:
        file1: Path to first parquet file
        file2: Path to second parquet file
        output: Path to save combined parquet file
        key_columns: List of columns to use for duplicate detection (None = all columns)
    """
    print(f"Loading {file1}...")
    df1 = pd.read_parquet(file1)
    print(f"  - Rows: {len(df1)}")
    
    print(f"Loading {file2}...")
    df2 = pd.read_parquet(file2)
    print(f"  - Rows: {len(df2)}")
    
    # Combine
    print("\nCombining datasets...")
    combined = pd.concat([df1, df2], ignore_index=True)
    print(f"  - Combined rows: {len(combined)}")
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    if key_columns:
        print(f"  - Using columns: {key_columns}")
        deduplicated = combined.drop_duplicates(subset=key_columns, keep='first')
    else:
        print("  - Using all columns")
        deduplicated = combined.drop_duplicates(keep='first')
    
    print(f"  - Rows after deduplication: {len(deduplicated)}")
    print(f"  - Duplicates removed: {len(combined) - len(deduplicated)}")
    
    # Save
    deduplicated.to_parquet(output, index=False)
    print(f"\nSaved to {output}")
    
    # Show sample statistics
    print("\nColumn names:")
    print(deduplicated.columns.tolist())
    print("\nFirst row sample:")
    print(deduplicated.head(1).to_dict('records')[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and deduplicate parquet files")
    parser.add_argument("--file1", required=True, help="First parquet file")
    parser.add_argument("--file2", required=True, help="Second parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--key_columns", nargs="+", default=None,
                        help="Columns to use for duplicate detection (e.g., --key_columns prompt response)")
    
    args = parser.parse_args()
    
    combine_and_deduplicate(args.file1, args.file2, args.output, args.key_columns)