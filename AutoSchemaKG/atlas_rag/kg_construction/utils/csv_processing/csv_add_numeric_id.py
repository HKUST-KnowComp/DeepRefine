import csv
from tqdm import tqdm

def check_created_csv_header(keyword, csv_dir):
    keyword_to_paths ={
        'cc_en':{
            'node_with_numeric_id': f"{csv_dir}/triple_nodes_cc_en_from_json_without_emb_with_numeric_id.csv",
            'edge_with_numeric_id': f"{csv_dir}/triple_edges_cc_en_from_json_without_emb_with_numeric_id.csv",
            'text_with_numeric_id': f"{csv_dir}/text_nodes_cc_en_from_json_with_numeric_id.csv",
            'concept_with_numeric_id': f"{csv_dir}/concept_nodes_pes2o_abstract_from_json_without_emb_with_numeric_id.csv",
        },
        'pes2o_abstract':{
            'node_with_numeric_id': f"{csv_dir}/triple_nodes_pes2o_abstract_from_json_without_emb_with_numeric_id.csv",
            'edge_with_numeric_id': f"{csv_dir}/triple_edges_pes2o_abstract_from_json_without_emb_full_concept_with_numeric_id.csv",
            'text_with_numeric_id': f"{csv_dir}/text_nodes_pes2o_abstract_from_json_with_numeric_id.csv",
        },
        'en_simple_wiki_v0':{
            'node_with_numeric_id': f"{csv_dir}/triple_nodes_en_simple_wiki_v0_from_json_without_emb_with_numeric_id.csv",
            'edge_with_numeric_id': f"{csv_dir}/triple_edges_en_simple_wiki_v0_from_json_without_emb_full_concept_with_numeric_id.csv",
            'text_with_numeric_id': f"{csv_dir}/text_nodes_en_simple_wiki_v0_from_json_with_numeric_id.csv",
        },
    }
    for key, path in keyword_to_paths[keyword].items():
        with open(path) as infile:
            reader = csv.reader(infile)
            header = next(reader)
            print(f"Header of {key}: {header}")
            
            # print first 5 rows
            for i, row in enumerate(reader):
                if i < 1:
                    print(row)
                else:
                    break

def add_csv_columns(node_csv, edge_csv, text_csv, node_with_numeric_id, edge_with_numeric_id, text_with_numeric_id):
    with open(node_csv) as infile, open(node_with_numeric_id, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header = next(reader)
        print(header)
        label_index = header.index(':LABEL')
        header.insert(label_index, 'numeric_id')  # Add new column name
        writer.writerow(header)
        for row_number, row in tqdm(enumerate(reader), desc="Adding numeric ID"):
            row.insert(label_index, row_number)  # Add numeric ID before ':LABEL'
            writer.writerow(row)
    with open(edge_csv) as infile, open(edge_with_numeric_id, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header = next(reader)
        print(header)
        label_index = header.index(':TYPE')
        header.insert(label_index, 'numeric_id')  # Add new column name
        writer.writerow(header)
        for row_number, row in tqdm(enumerate(reader), desc="Adding numeric ID"):
            row.insert(label_index, row_number)  # Add numeric ID before ':LABEL'
            writer.writerow(row)
    with open(text_csv) as infile, open(text_with_numeric_id, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header = next(reader)
        print(header)
        label_index = header.index(':LABEL')
        header.insert(label_index, 'numeric_id')  # Add new column name
        writer.writerow(header)
        for row_number, row in tqdm(enumerate(reader), desc="Adding numeric ID"):
            row.insert(label_index, row_number)  # Add numeric ID before ':LABEL'
            writer.writerow(row)
            

# def add_csv_columns(keyword, csv_dir):
#     keyword_to_paths ={
#         'cc_en':{
#             'node_csv': f"{csv_dir}/triple_nodes_cc_en_from_json_without_emb.csv",
#             'edge_csv': f"{csv_dir}/triple_edges_cc_en_from_json_without_emb.csv",
#             'text_csv': f"{csv_dir}/text_nodes_cc_en_from_json.csv",
            
#             'node_with_numeric_id': f"{csv_dir}/triple_nodes_cc_en_from_json_without_emb_with_numeric_id.csv",
#             'edge_with_numeric_id': f"{csv_dir}/triple_edges_cc_en_from_json_without_emb_with_numeric_id.csv",
#             'text_with_numeric_id': f"{csv_dir}/text_nodes_cc_en_from_json_with_numeric_id.csv"
#         },
#         'pes2o_abstract':{
#             'node_csv': f"{csv_dir}/triple_nodes_pes2o_abstract_from_json_without_emb.csv",
#             'edge_csv': f"{csv_dir}/triple_edges_pes2o_abstract_from_json_without_emb_full_concept.csv",
#             'text_csv': f"{csv_dir}/text_nodes_pes2o_abstract_from_json.csv",
            
#             'node_with_numeric_id': f"{csv_dir}/triple_nodes_pes2o_abstract_from_json_without_emb_with_numeric_id.csv",
#             'edge_with_numeric_id': f"{csv_dir}/triple_edges_pes2o_abstract_from_json_without_emb_full_concept_with_numeric_id.csv",
#             'text_with_numeric_id': f"{csv_dir}/text_nodes_pes2o_abstract_from_json_with_numeric_id.csv"
#         },
#         'en_simple_wiki_v0':{
#             'node_csv': f"{csv_dir}/triple_nodes_en_simple_wiki_v0_from_json_without_emb.csv",
#             'edge_csv': f"{csv_dir}/triple_edges_en_simple_wiki_v0_from_json_without_emb_full_concept.csv",
#             'text_csv': f"{csv_dir}/text_nodes_en_simple_wiki_v0_from_json.csv",
            
#             'node_with_numeric_id': f"{csv_dir}/triple_nodes_en_simple_wiki_v0_from_json_without_emb_with_numeric_id.csv",
#             'edge_with_numeric_id': f"{csv_dir}/triple_edges_en_simple_wiki_v0_from_json_without_emb_full_concept_with_numeric_id.csv",
#             'text_with_numeric_id': f"{csv_dir}/text_nodes_en_simple_wiki_v0_from_json_with_numeric_id.csv"
#         },
#     }
#     # ouput node
#     with open(keyword_to_paths[keyword]['node_csv']) as infile, open(keyword_to_paths[keyword]['node_with_numeric_id'], 'w') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)

#         # Read the header
#         header = next(reader)
#         print(header)
#         # Insert 'numeric_id' before ':LABEL'
#         label_index = header.index(':LABEL')
#         header.insert(label_index, 'numeric_id')  # Add new column name
#         writer.writerow(header)

#         # Process each row and add a numeric ID
#         for row_number, row in tqdm(enumerate(reader), desc="Adding numeric ID"):
#             row.insert(label_index, row_number)  # Add numeric ID before ':LABEL'
#             writer.writerow(row)
            
#     # output edge (TYPE instead of LABEL for edge)
#     with open(keyword_to_paths[keyword]['edge_csv']) as infile, open(keyword_to_paths[keyword]['edge_with_numeric_id'], 'w') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)

#         # Read the header
#         header = next(reader)
#         print(header)
#         # Insert 'numeric_id' before ':TYPE'
#         label_index = header.index(':TYPE')
#         header.insert(label_index, 'numeric_id')  # Add new column name
#         writer.writerow(header)

#         # Process each row and add a numeric ID
#         for row_number, row in tqdm(enumerate(reader), desc="Adding numeric ID"):
#             row.insert(label_index, row_number)  # Add numeric ID before ':LABEL'
#             writer.writerow(row)
            
#     # output text
#     with open(keyword_to_paths[keyword]['text_csv']) as infile, open(keyword_to_paths[keyword]['text_with_numeric_id'], 'w') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)

#         # Read the header
#         header = next(reader)
#         print(header)
#         # Insert 'numeric_id' before ':LABEL'
#         label_index = header.index(':LABEL')
#         header.insert(label_index, 'numeric_id')  # Add new column name
#         writer.writerow(header)

#         # Process each row and add a numeric ID
#         for row_number, row in tqdm(enumerate(reader), desc="Adding numeric ID"):
#             row.insert(label_index, row_number)  # Add numeric ID before ':LABEL'
#             writer.writerow(row)

if __name__ == "__main__":
    keyword = "en_simple_wiki_v0"
    csv_dir = "./import"  # Change this to your CSV directory
    add_csv_columns(keyword, csv_dir)
    # check_created_csv_header(keyword)