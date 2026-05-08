import networkx as nx
import numpy as np
from collections import Counter
import math

def analyze_graph_properties(graph_path, max_hop=3):
    # Load directed graph
    G = nx.read_graphml(graph_path)
    if not nx.is_directed(G):
        G = G.to_directed()

    results = {}

    # --- Node-level ---
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    results["avg_in_degree"] = np.mean(list(in_degrees.values()))
    results["avg_out_degree"] = np.mean(list(out_degrees.values()))
    results["max_in_degree"] = np.max(list(in_degrees.values()))
    results["max_out_degree"] = np.max(list(out_degrees.values()))

    # Entity coverage (approx: distinct nodes / total nodes in docs → assume all nodes = entity mentions)
    results["entity_count"] = G.number_of_nodes()

    # Relation diversity per node
    rel_counts = []
    for n in G.nodes():
        rels = set([G.edges[e]["relation"] for e in G.out_edges(n)])
        rel_counts.append(len(rels))
    results["avg_relation_diversity"] = np.mean(rel_counts)

    # --- Edge-level ---
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    results["edge_density"] = num_edges / (num_nodes * (num_nodes - 1))

    reciprocal = 0
    for u, v in G.edges():
        if G.has_edge(v, u):
            reciprocal += 1
    results["reciprocal_edge_fraction"] = reciprocal / num_edges

    # --- Path/Subgraph ---
    if nx.is_weakly_connected(G):
        sp_lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=max_hop))
        lengths = []
        for d in sp_lengths.values():
            lengths.extend(list(d.values()))
        results["avg_shortest_path_length"] = np.mean(lengths)
    else:
        results["avg_shortest_path_length"] = None

    results["num_cycles"] = sum(1 for _ in nx.simple_cycles(G))

    # --- Graph-level ---
    results["num_connected_components"] = nx.number_weakly_connected_components(G)
    if nx.is_weakly_connected(G):
        results["diameter"] = nx.diameter(G.to_undirected())
    else:
        results["diameter"] = None

    # Directed clustering coefficient
    try:
        results["avg_clustering_coefficient"] = nx.average_clustering(G.to_undirected())
    except Exception:
        results["avg_clustering_coefficient"] = None

    # Relation entropy
    relation_list = [data["relation"] for _, _, data in G.edges(data=True) if "relation" in data]
    rel_freq = np.array(list(Counter(relation_list).values()))
    probs = rel_freq / rel_freq.sum()
    results["relation_entropy"] = -np.sum(probs * np.log(probs + 1e-9))

    return results


if __name__ == "__main__":
    graph_path = "kg.graphml"  # replace with your path
    props = analyze_graph_properties(graph_path)
    for k, v in props.items():
        print(f"{k}: {v}")
