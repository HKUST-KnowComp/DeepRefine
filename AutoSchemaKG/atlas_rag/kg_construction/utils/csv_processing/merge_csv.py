import os
import glob

def merge_csv_files(output_file, input_dir):
    """
    Merge all CSV files in the input directory into a single output file.

    Args:
        output_file (str): Path to the output CSV file.
        input_dir (str): Directory containing the input CSV files.
    """
    # Delete the output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Write the header to the output file
    with open(output_file, 'w') as outfile:
        outfile.write("node,conceptualized_node,node_type\n")

    # Append the contents of all CSV files in the input directory
    for csv_file in glob.glob(os.path.join(input_dir, '*.csv')):
        with open(csv_file, 'r') as infile:
            # Skip the header line
            next(infile)
            # Append the remaining lines to the output file
            with open(output_file, 'a') as outfile:
                outfile.writelines(infile)
