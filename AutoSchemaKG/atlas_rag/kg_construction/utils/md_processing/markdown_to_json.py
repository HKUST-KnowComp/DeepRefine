import argparse
import json
import os
import sys
from pathlib import Path

# Set up argument parser
parser = argparse.ArgumentParser(description="Convert all Markdown files in a folder to separate JSON files.")
parser.add_argument(
    "--input", required=True, help="Path to the folder containing Markdown files"
)
parser.add_argument(
    "--output", default=None, help="Output folder for JSON files (defaults to input folder if not specified)"
)

# Parse arguments
args = parser.parse_args()

# Resolve input folder path
input_folder = Path(args.input)
if not input_folder.is_dir():
    print(f"Error: '{args.input}' is not a directory.", file=sys.stderr)
    sys.exit(1)

# Set output folder (use input folder if not specified)
output_folder = Path(args.output) if args.output else input_folder
output_folder.mkdir(parents=True, exist_ok=True)

# Find all .md files in the input folder
markdown_files = [f for f in input_folder.iterdir() if f.suffix.lower() == ".md"]

if not markdown_files:
    print(f"Error: No Markdown files found in '{args.input}'.", file=sys.stderr)
    sys.exit(1)

# Process each Markdown file
for file in markdown_files:
    try:
        # Read the content of the file
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        # Create the JSON object
        obj = {
            "id": "1",
            "text": content,
            "metadata": {
                "lang": "en"
            }
        }

        # Create output JSON filename (e.g., file1.md -> file1.json)
        output_file = output_folder / f"{file.stem}.json"

        # Write JSON to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([obj], f, indent=4)

        print(f"Successfully converted '{file}' to '{output_file}'")
    except FileNotFoundError:
        print(f"Error: File '{file}' not found.", file=sys.stderr)
    except Exception as e:
        print(f"Error processing file '{file}': {e}", file=sys.stderr)