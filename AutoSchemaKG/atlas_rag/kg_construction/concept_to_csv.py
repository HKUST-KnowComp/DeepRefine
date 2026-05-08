import ast
import uuid
import csv
from tqdm import tqdm
import hashlib
import os

def generate_uuid():
    """Generate a random UUID"""
    return str(uuid.uuid4())

def parse_concepts(s):
    """Parse concepts field and filter empty values"""
    try:
        parsed = ast.literal_eval(s) if s and s != '[]' else []
        return [c.strip() for c in parsed if c.strip()]
    except:
        return []
    

# Function to compute a hash ID from text
def compute_hash_id(text):
    # Use SHA-256 to generate a hash
    text = text + '_concept'
    hash_object = hashlib.sha256(text.encode('utf-8'))
    return hash_object.hexdigest()  # Return hash as a hex string


def all_concept_triples_csv_to_csv(node_file, edge_file, concepts_file, output_node_file, output_edge_file, output_full_concept_triple_edges):

    # to deal add output the concepts nodes, edges, and new full_triple_edges, 
    # we need to read the concepts maps to the memory, as it is usually not too large.
    # Then we need to iterate over the triple nodes to create concept edges 
    # Finally we iterate over the triple edges to create the full_triple_edges
     
    # Read missing concept
    # relation_concepts_mapping = {}
    # all_missing_concepts = []
    
    # check if all output directories exist
    output_dir = os.path.dirname(output_node_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.dirname(output_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.dirname(output_full_concept_triple_edges)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    node_to_concepts = {}
    relation_to_concepts = {}

    all_concepts = set()
    with open(concepts_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Load missing concepts list
        print("Loading concepts...")
        for row in tqdm(reader):


            if row['node_type'] == 'relation':
                relation = row['node']
                concepts = [c.strip() for c in row['conceptualized_node'].split(',') if c.strip()]

                if relation not in relation_to_concepts:
                    relation_to_concepts[relation] = concepts
                else:
                    relation_to_concepts[relation].extend(concepts)
                    relation_to_concepts[relation] = list(set(relation_to_concepts[relation]))
                
            else:
                node = row['node']
                concepts = [c.strip() for c in row['conceptualized_node'].split(',') if c.strip()]

                if node not in node_to_concepts:
                    node_to_concepts[node] = concepts
                else:
                    node_to_concepts[node].extend(concepts)
                    node_to_concepts[node] = list(set(node_to_concepts[node]))

    print("Loading concepts done.")
    print(f"Relation to concepts: {len(relation_to_concepts)}")
    print(f"Node to concepts: {len(node_to_concepts)}")

    # Read triple nodes and write to output concept edges files
    print("Processing triple nodes...")
    with open(node_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # name:ID,type,concepts,synsets,:LABEL
        header = next(reader)
        
        with open (output_edge_file, 'w', newline='', encoding='utf-8') as f_out:
            
            writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
            writer.writerow([':START_ID', ':END_ID', 'relation', ':TYPE'])
            
            for row in tqdm(reader):
                node_name = row[0]
                if node_name in node_to_concepts:
                   
                    for concept in node_to_concepts[node_name]:
                        concept_id = compute_hash_id(concept)
                        writer.writerow([row[0], concept_id, 'has_concept', 'Concept'])
                        all_concepts.add(concept)
                    

                for concept in parse_concepts(row[2]):
                    concept_id = compute_hash_id(concept)
                    writer.writerow([row[0], concept_id, 'has_concept', 'Concept'])
                    all_concepts.add(concept)

    # Read the concept nodes and write to output concept nodes file
    print("Processing concept nodes...")
    with open (output_node_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow(['concept_id:ID', 'name', ':LABEL'])

        for concept in tqdm(all_concepts):
            concept_id = compute_hash_id(concept)
            writer.writerow([concept_id, concept, 'Concept'])
           
    
    # Read triple edges and write to output full concept triple edges file
    print("Processing triple edges...")
    with open(edge_file, 'r', encoding='utf-8') as f:
        with open(output_full_concept_triple_edges, 'w', newline='', encoding='utf-8') as f_out:
            reader = csv.reader(f)
            writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)


            header = next(reader)
            writer.writerow([':START_ID', ':END_ID', 'relation', 'concepts', 'synsets', ':TYPE'])

            for row in tqdm(reader):
                src_id = row[0]
                end_id = row[1]
                relation = row[2]
                concepts = row[3]
                synsets = row[4]

                original_concepts = parse_concepts(concepts)
             

                if relation in relation_to_concepts:
                    for concept in relation_to_concepts[relation]:
                        if concept not in original_concepts:
                            original_concepts.append(concept)
                            original_concepts = list(set(original_concepts))
                
                writer.writerow([src_id, end_id, relation, original_concepts, synsets, 'Relation'])
    return