"""
Script to add prompt and response columns to existing SFT parquet files
without re-running the extraction process.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np


SYSTEM_PROMPT = """You are an expert knowledge graph constructor.  
Your task is to extract factual information from the provided text and represent it strictly as a JSON array of knowledge graph triples.  

### Output Format
- The output must be a **JSON array**.
- Each element in the array must be a **JSON object** with exactly three non-empty keys:
  - "subject": the main entity, concept, event, or attribute.  
  - "relation": a concise, descriptive phrase or verb that describes the relationship (e.g., "founded by", "started on", "is a", "has circulation of").  
  - "object": the entity, concept, value, event, or attribute that the subject has a relationship with.  

### Constraints
- **Do not include any text other than the JSON output.**
- Do not add explanations, comments, or formatting outside of the JSON array.  
- Extract **all possible and relevant triples**.  
- All keys must exist and all values must be non-empty strings.  
- The "subject" and "object" can be specific entities (e.g., "Radio City", "Football in Albania", "Echosmith") or specific values (e.g., "3 July 2001", "1,310,696").  
- If no triples can be extracted, return exactly: `[]`."""


def format_context(full_context) -> str:
    """Format full_context list into a string"""
    # Handle numpy arrays
    if isinstance(full_context, np.ndarray):
        full_context = full_context.tolist()
    
    if isinstance(full_context, list):
        return "\n\n".join([f"{doc}" for i, doc in enumerate(full_context)])
    return str(full_context)


def format_triples_as_json(triples) -> str:
    """Format triples as JSON string for response"""
    # Handle numpy arrays
    if isinstance(triples, np.ndarray):
        triples = triples.tolist()
    
    # Handle None or empty
    if triples is None or len(triples) == 0:
        return "[]"
    
    # Convert any numpy types within the triples
    clean_triples = []
    for triple in triples:
        if isinstance(triple, dict):
            clean_triple = {
                k: str(v) if isinstance(v, (np.integer, np.floating)) else v 
                for k, v in triple.items()
            }
            clean_triples.append(clean_triple)
        else:
            clean_triples.append(triple)
    
    return json.dumps(clean_triples, ensure_ascii=False)


def create_prompt(full_context) -> str:
    """Create the full prompt with system prompt and user context"""
    context_text = format_context(full_context)
    user_prompt = f"Extract from the documents:\n\n{context_text}"
    
    # Format as chat-style prompt
    full_prompt = f"""{SYSTEM_PROMPT}\n\n
{user_prompt}"""
    return full_prompt


def add_prompt_response_columns(
    input_path: str,
    output_path: str,
    keep_all_columns: bool = False
):
    """
    Add prompt and response columns to existing parquet file
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save modified parquet file
        keep_all_columns: If True, keep all original columns. If False, only keep prompt and response.
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Processing {len(df)} rows...")
    
    # Add prompt column
    print("Creating prompts...")
    df['prompt'] = df['full_context'].apply(create_prompt)
    
    # Add response column (triples as JSON string)
    print("Formatting responses...")
    df['response'] = df['extracted_triples'].apply(format_triples_as_json)
    
    # Optionally keep only prompt and response columns
    if not keep_all_columns:
        df = df[['prompt', 'response']]
    
    # Save modified parquet
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    
    # Print sample
    print("\n" + "="*80)
    print("SAMPLE ROW:")
    print("="*80)
    print("\nPROMPT (first 500 chars):")
    print(df['prompt'].iloc[0][:500])
    print("\n...")
    print("\nRESPONSE (first 500 chars):")
    print(df['response'].iloc[0][:500])
    print("\n...")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS:")
    print("="*80)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Avg prompt length: {df['prompt'].str.len().mean():.0f} chars")
    print(f"Avg response length: {df['response'].str.len().mean():.0f} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Add prompt and response columns to existing SFT parquet files"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save modified parquet file"
    )
    parser.add_argument(
        "--keep-all",
        "-k",
        action="store_true",
        help="Keep all original columns (default: only keep prompt and response)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    add_prompt_response_columns(
        input_path=args.input,
        output_path=args.output,
        keep_all_columns=args.keep_all
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()