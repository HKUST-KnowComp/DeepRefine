import faiss
from neo4j import Driver
import time
from graphdatascience import GraphDataScience
from atlas_rag.retriever.lkg_retriever.base import BaseLargeKGRetriever

def build_projection_graph(driver: GraphDataScience):
    project_graph_1 = "largekgrag_graph"
    is_project_graph_1_exist = False
    # is_project_graph_2_exist = False
    result = driver.graph.list()
    for index, row in result.iterrows():
        if row['graphName'] == project_graph_1:
            is_project_graph_1_exist = True
        # if row['graphName'] == project_graph_2:
        #     is_project_graph_2_exist = True
    
    if not is_project_graph_1_exist:
        start_time = time.time()
        node_properties = ["Node"]
        relation_projection = [ "Relation"]
        result = driver.graph.project(
            project_graph_1,
            node_properties,
            relation_projection
        )
        graph = driver.graph.get(project_graph_1)
        print(f"Projection graph {project_graph_1} created in {time.time() - start_time:.2f} seconds")

def build_neo4j_label_index(driver: GraphDataScience):
    with driver.session() as session:
        index_name = f"NodeNumericIDIndex"
        # Check if the index already exists
        existing_indexes = session.run("SHOW INDEXES").data()
        index_exists = any(index['name'] == index_name for index in existing_indexes)
        # Drop the index if it exists
        if not index_exists:
            start_time = time.time()
            session.run(f"CREATE INDEX {index_name} FOR (n:Node) ON (n.numeric_id)")
            print(f"Index {index_name} created in {time.time() - start_time:.2f} seconds")
            
        index_name = f"TextNumericIDIndex"
        index_exists = any(index['name'] == index_name for index in existing_indexes)
        if not index_exists:
            start_time = time.time()
            session.run(f"CREATE INDEX {index_name} FOR (t:Text) ON (t.numeric_id)")
            print(f"Index {index_name} created in {time.time() - start_time:.2f} seconds")
        
        index_name = f"EntityEventEdgeNumericIDIndex"
        index_exists = any(index['name'] == index_name for index in existing_indexes)
        if not index_exists:
            start_time = time.time()
            session.run(f"CREATE INDEX {index_name} FOR ()-[r:Relation]-() on (r.numeric_id)")
            print(f"Index {index_name} created in {time.time() - start_time:.2f} seconds")

def load_indexes(path_dict):
    for key, value in path_dict.items():
        if key == 'node':
            node_index = faiss.read_index(value, faiss.IO_FLAG_MMAP)
            print(f"Node index loaded from {value}")
        elif key == 'edge':
            edge_index = faiss.read_index(value, faiss.IO_FLAG_MMAP)
            print(f"Edge index loaded from {value}")
        elif key == 'text':
            passage_index = faiss.read_index(value, faiss.IO_FLAG_MMAP)
            print(f"Passage index loaded from {value}")
    return node_index, edge_index, passage_index

def start_up_large_kg_index_graph(neo4j_driver: Driver)->BaseLargeKGRetriever:    
    gds_driver = GraphDataScience(neo4j_driver)
    # build label index and projection graph
    build_neo4j_label_index(neo4j_driver)
    build_projection_graph(gds_driver)
