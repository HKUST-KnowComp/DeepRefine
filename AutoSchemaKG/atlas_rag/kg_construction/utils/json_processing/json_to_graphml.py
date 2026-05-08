import networkx as nx
import json
from tqdm import tqdm
import os
import hashlib

def get_node_id(entity_name, entity_to_id):
    """Returns existing or creates new nX ID for an entity using a hash-based approach."""
    if entity_name not in entity_to_id:
        # Use a hash function to generate a unique ID
        hash_object = hashlib.md5(entity_name.encode())  # Use MD5 or another hashing algorithm
        hash_hex = hash_object.hexdigest()  # Get the hexadecimal representation of the hash
        # Use the first 8 characters of the hash as the ID (you can adjust the length as needed)
        entity_to_id[entity_name] = f'n{hash_hex[:16]}'
    return entity_to_id[entity_name]

def clean_text(text):
    # remove NUL as well
    new_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\v", " ").replace("\f", " ").replace("\b", " ").replace("\a", " ").replace("\e", " ").replace(";", ",")
    new_text = new_text.replace("\x00", "")
    return new_text


def process_kg_data(input_passage_dir, input_triple_dir, output_dir, keyword):
    # Get file names containing the keyword
    file_names = [file for file in list(os.listdir(input_triple_dir)) if keyword in file]
    print(f"Keyword: {keyword}")
    print(f"Number of files: {len(file_names)}")
    print(file_names)

    passage_file_names = [file for file in list(os.listdir(input_passage_dir)) if keyword in file]
    print(f'Passage file names: {passage_file_names}')

    g = nx.DiGraph()
    print("Graph created.")
    entity_to_id = {}
    # check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory {output_dir} created.")
        
    output_path = f"{output_dir}/{keyword}_kg_from_corpus.graphml"

    # Create the original_text to node_id dictionary and add passage node to the graph
    with open(f"{input_passage_dir}/{passage_file_names[0]}") as f:
        data = json.load(f)
        for item in tqdm(data, desc="Processing passages"):
            passage_id = item["id"]
            passage_text = item["text"]
            node_id = get_node_id(passage_text, entity_to_id)
            if passage_text.isspace() or len(passage_text) == 0:
                continue
            # Add the passage node to the graph
            g.add_node(node_id, type="passage", id=passage_text, file_id=passage_id)

    for file_name in tqdm(file_names):
        print(f"Processing {file_name}")
        input_file_path = f"{input_triple_dir}/{file_name}"
        with open(input_file_path) as f:
            for line in tqdm(f):
                data = json.loads(line)
                metadata = data["metadata"]
                file_id = data["id"]
                original_text = data["original_text"]
                entity_relation_dict = data["entity_relation_dict"]
                event_entity_relation_dict = data["event_entity_relation_dict"]
                event_relation_dict = data["event_relation_dict"]

                # Process entity triples
                entity_triples = []
                for entity_triple in entity_relation_dict:
                    if not all(key in entity_triple for key in ["Head", "Relation", "Tail"]):
                        continue
                    head_entity = clean_text(entity_triple["Head"])
                    relation = clean_text(entity_triple["Relation"])
                    tail_entity = clean_text(entity_triple["Tail"])
                    if head_entity.isspace() or len(head_entity) == 0 or tail_entity.isspace() or len(tail_entity) == 0:
                        continue
                    entity_triples.append((head_entity, relation, tail_entity))

                # Add entity triples to the graph
                for triple in entity_triples:
                    head_id = get_node_id(triple[0], entity_to_id)
                    tail_id = get_node_id(triple[2], entity_to_id)
                    g.add_node(head_id, type="entity", id=triple[0])
                    g.add_node(tail_id, type="entity", id=triple[2])
                    g.add_edge(head_id, get_node_id(original_text, entity_to_id), relation='mention in')
                    g.add_edge(tail_id, get_node_id(original_text, entity_to_id), relation='mention in')
                    g.add_edge(head_id, tail_id, relation=triple[1])
                    for node_id in [head_id, tail_id]:
                        if "file_id" not in g.nodes[node_id]:
                            g.nodes[node_id]["file_id"] = str(file_id)
                        else:
                            g.nodes[node_id]["file_id"] += "," + str(file_id)
                    edge = g.edges[head_id, tail_id]
                    if "file_id" not in edge:
                        edge["file_id"] = str(file_id)
                    else:
                        edge["file_id"] += "," + str(file_id)

                # Process event triples
                event_triples = []
                for event_triple in event_relation_dict:
                    if not all(key in event_triple for key in ["Head", "Relation", "Tail"]):
                        continue
                    head_event = clean_text(event_triple["Head"])
                    relation = clean_text(event_triple["Relation"])
                    tail_event = clean_text(event_triple["Tail"])
                    if head_event.isspace() or len(head_event) == 0 or tail_event.isspace() or len(tail_event) == 0:
                        continue
                    event_triples.append((head_event, relation, tail_event))

                # Add event triples to the graph
                for triple in event_triples:
                    head_id = get_node_id(triple[0], entity_to_id)
                    tail_id = get_node_id(triple[2], entity_to_id)
                    g.add_node(head_id, type="event", id=triple[0])
                    g.add_node(tail_id, type="event", id=triple[2])
                    g.add_edge(head_id, get_node_id(original_text, entity_to_id), relation='mention in')
                    g.add_edge(tail_id, get_node_id(original_text, entity_to_id), relation='mention in')
                    g.add_edge(head_id, tail_id, relation=triple[1])
                    for node_id in [head_id, tail_id]:
                        if "file_id" not in g.nodes[node_id]:
                            g.nodes[node_id]["file_id"] = str(file_id)
                        else:
                            g.nodes[node_id]["file_id"] += "," + str(file_id)
                    edge = g.edges[head_id, tail_id]
                    if "file_id" not in edge:
                        edge["file_id"] = str(file_id)
                    else:
                        edge["file_id"] += "," + str(file_id)

                # Process event-entity triples
                event_entity_triples = []
                for event_entity_participations in event_entity_relation_dict:
                    if not all(key in event_entity_participations for key in ["Event", "Entity"]):
                        continue
                    event = clean_text(event_entity_participations["Event"])
                    if event.isspace() or len(event) == 0:
                        continue
                    for entity in event_entity_participations["Entity"]:
                        if not isinstance(entity, str) or entity.isspace() or len(entity) == 0:
                            continue
                        entity = clean_text(entity)
                        event_entity_triples.append((event, "is participated by", entity))

                # Add event-entity triples to the graph
                for triple in event_entity_triples:
                    head_id = get_node_id(triple[0], entity_to_id)
                    tail_id = get_node_id(triple[2], entity_to_id)
                    g.add_node(head_id, type="event", id=triple[0])
                    g.add_node(tail_id, type="entity", id=triple[2])
                    g.add_edge(head_id, tail_id, relation=triple[1])
                    for node_id in [head_id, tail_id]:
                        if "file_id" not in g.nodes[node_id]:
                            g.nodes[node_id]["file_id"] = str(file_id)
                    edge = g.edges[head_id, tail_id]
                    if "file_id" not in edge:
                        edge["file_id"] = str(file_id)
                    else:
                        edge["file_id"] += "," + str(file_id)

    print(f"Number of nodes: {g.number_of_nodes()}")
    print(f"Number of edges: {g.number_of_edges()}")
    print(f"Graph density: {nx.density(g)}")
    with open(output_path, 'wb') as f:
        nx.write_graphml(g, f, infer_numeric_types=True)


