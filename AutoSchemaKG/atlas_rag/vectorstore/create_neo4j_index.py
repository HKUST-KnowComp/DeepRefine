import faiss
import numpy as np
import time
import logging
from atlas_rag.kg_construction.utils.csv_processing.csv_to_npy import convert_csv_to_npy
def create_faiss_index(output_directory, filename_pattern, index_type="HNSW,Flat", faiss_gpu = True):
        """
        Create faiss index for the graph, for index type, see https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

        "IVF65536_HNSW32,Flat" for 1M to 10M nodes

        "HNSW,Flat" for toy dataset

        """
        # Convert csv to npy
        convert_csv_to_npy(
            csv_path=f"{output_directory}/triples_csv/triple_nodes_{filename_pattern}_from_json_with_emb.csv",
            npy_path=f"{output_directory}/vector_index/triple_nodes_{filename_pattern}_from_json_with_emb.npy",
        )

        convert_csv_to_npy(
            csv_path=f"{output_directory}/triples_csv/text_nodes_{filename_pattern}_from_json_with_emb.csv",
            npy_path=f"{output_directory}/vector_index/text_nodes_{filename_pattern}_from_json_with_emb.npy",
        )

        convert_csv_to_npy(
            csv_path=f"{output_directory}/triples_csv/triple_edges_{filename_pattern}_from_json_with_concept_with_emb.csv",
            npy_path=f"{output_directory}/vector_index/triple_edges_{filename_pattern}_from_json_with_concept_with_emb.npy",
        )

        build_faiss_from_npy(
            index_type=index_type,
            index_path=f"{output_directory}/vector_index/triple_nodes_{filename_pattern}_from_json_with_emb_non_norm.index",
            npy_path=f"{output_directory}/vector_index/triple_nodes_{filename_pattern}_from_json_with_emb.npy",
            faiss_gpu=faiss_gpu
        )

        build_faiss_from_npy(
            index_type=index_type,
            index_path=f"{output_directory}/vector_index/text_nodes_{filename_pattern}_from_json_with_emb_non_norm.index",
            npy_path=f"{output_directory}/vector_index/text_nodes_{filename_pattern}_from_json_with_emb.npy",
            faiss_gpu=faiss_gpu
        )

        build_faiss_from_npy(
            index_type=index_type,
            index_path=f"{output_directory}/vector_index/triple_edges_{filename_pattern}_from_json_with_concept_with_emb_non_norm.index",
            npy_path=f"{output_directory}/vector_index/triple_edges_{filename_pattern}_from_json_with_concept_with_emb.npy",
            faiss_gpu=faiss_gpu
        )

# cannot avoid loading into memory when training
# simply try load all to train
def build_faiss_from_npy(index_type, index_path, npy_path, faiss_gpu = True):
    # check npy size.
    # shapes = []
    start_time = time.time()
    # with open(npy_path, "rb") as f:
    #     while True:
    #         try:
    #             array = np.load(f)
    #             shapes.append(array.shape)
    #         except Exception as e:
    #             print(f"Stopped loading due to: {str(e)}")
    #             break
    # if shapes:
    #     total_rows = sum(shape[0] for shape in shapes)
    #     dimension = shapes[0][1]
    #     print(f"Total embeddings in {npy_path}\n {total_rows}, Dimension: {dimension}")
    # minilm is 32
    # get the dimension from the npy file
    with open(npy_path, "rb") as f:
        array = np.load(f)
        dimension = array.shape[1]
    print(f"Dimension: {dimension}")
    index = faiss.index_factory(dimension, index_type,  faiss.METRIC_INNER_PRODUCT)

    if index_type.startswith("IVF"):
        index_ivf = faiss.extract_index_ivf(index)
        if faiss_gpu:
            clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        else:
            clustering_index = faiss.IndexFlatL2(index_ivf.d)
        index_ivf.clustering_index = clustering_index
    
        # Load data to match the training samples size.
        # Done by random picking indexes from shapes and check if the sum of the indexes is over the sample size or not. 
        # If yes then read them and start training, skip the np.load part for non chosen indexes
        
        # selected_indices = set()
        # possible_indices = list(range(len(shapes)))
        # selected_training_samples = 0
        # while selected_training_samples < max_training_samples and possible_indices:
        #     idx = random.choice(possible_indices)
        #     selected_indices.add(idx)
        #     selected_training_samples += shapes[idx][0]
        #     possible_indices.remove(idx)
        # print(f"Selected total: {selected_training_samples} samples for training")
    
        xt = []
        current_index = 0
        with open(npy_path, "rb") as f:
            while True:
                try:
                    # array = np.load(f)
                    # if current_index in selected_indices:
                    array = np.load(f)
                    # faiss.normalize_L2(array)
                    xt.append(array)
                    # current_index += 1
                except Exception as e:
                    logging.info(f"Stopped loading due to: {str(e)}")
                    break
            if xt:
                xt = np.vstack(xt)
        logging.info(f"Loading time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        index.train(xt)
        end_time = time.time()
        logging.info(f"Training time: {end_time - start_time:.2f} seconds")
        del xt
    start_time = time.time()
    with open(npy_path, "rb") as f:
        while True:
            try:
                array = np.load(f)
                # faiss.normalize_L2(array)
                index.add(array)
            except Exception as e:
                logging.info(f"Stopped loading due to: {str(e)}")
                break
    logging.info(f"Adding time: {time.time() - start_time:.2f} seconds")
    
    # Convert the GPU index to a CPU index for saving
    if faiss_gpu:
        index = faiss.index_gpu_to_cpu(index)

    # Save the CPU index to a file
    faiss.write_index(index, index_path)

def train_and_write_indexes(keyword, npy_dir="./import"):
    keyword_to_paths = {
        'cc_en': {
            'npy':{
                'node': f"{npy_dir}/triple_nodes_cc_en_from_json_2.npy",
                # 'edge': f"{npy_dir}/triple_edges_cc_en_from_json_2.npy",
                'text': f"{npy_dir}/text_nodes_cc_en_from_json_with_emb_2.npy",
            },
            'index':{
                'node': f"{npy_dir}/triple_nodes_cc_en_from_json_non_norm.index",
                # 'edge': f"{npy_dir}/triple_edges_cc_en_from_json_non_norm.index",
                'text': f"{npy_dir}/text_nodes_cc_en_from_json_with_emb_non_norm.index",
            },
            'index_type':{
                'node': "IVF1048576_HNSW32,Flat",
                # 'edge': "IVF1048576_HNSW32,Flat",
                'text': "IVF262144_HNSW32,Flat",
            },
            'csv':{
                'node': f"{npy_dir}/triple_nodes_cc_en_from_json.csv",
                # 'edge': ff"{npy_dir}/triple_edges_cc_en_from_json.csv",
                'text': f"{npy_dir}/text_nodes_cc_en_from_json_with_emb.csv",
            }
        },
        'pes2o_abstract': {
            'npy':{
                'node': f"{npy_dir}/triple_nodes_pes2o_abstract_from_json.npy",
                # 'edge': f"{npy_dir}/triple_edges_pes2o_abstract_from_json.npy",
                'text': f"{npy_dir}/text_nodes_pes2o_abstract_from_json_with_emb.npy",
            },
            'index':{
                'node': f"{npy_dir}/triple_nodes_pes2o_abstract_from_json_non_norm.index",
                # 'edge': f"{npy_dir}/triple_edges_pes2o_abstract_from_json_non_norm.index",
                'text': f"{npy_dir}/text_nodes_pes2o_abstract_from_json_with_emb_non_norm.index",
            },
            'index_type':{
                'node': "IVF1048576_HNSW32,Flat",
                # 'edge': "IVF1048576_HNSW32,Flat",
                'text': "IVF65536_HNSW32,Flat",
            },
            'csv':{
                'node_csv': f"{npy_dir}/triple_nodes_pes2o_abstract_from_json.csv",
                # 'edge_csv': ff"{npy_dir}/triple_edges_pes2o_abstract_from_json.csv",
                'text_csv': f"{npy_dir}/text_nodes_pes2o_abstract_from_json_with_emb.csv",
            }
        },
        'en_simple_wiki_v0': {
            'npy':{
                'node': f"{npy_dir}/triple_nodes_en_simple_wiki_v0_from_json.npy",
                # 'edge': f"{npy_dir}/triple_edges_en_simple_wiki_v0_from_json.npy",
                'text': f"{npy_dir}/text_nodes_en_simple_wiki_v0_from_json_with_emb.npy",
            },
            'index':{
                'node': f"{npy_dir}/triple_nodes_en_simple_wiki_v0_from_json_non_norm.index",
                # 'edge': f"{npy_dir}/triple_edges_en_simple_wiki_v0_from_json_non_norm.index",
                'text': f"{npy_dir}/text_nodes_en_simple_wiki_v0_from_json_with_emb_non_norm.index",
            },
            'index_type':{
                'node': "IVF1048576_HNSW32,Flat",
                # 'edge': "IVF1048576_HNSW32,Flat",
                'text': "IVF65536_HNSW32,Flat",
            },
            'csv':{
                'node_csv': f"{npy_dir}/triple_nodes_en_simple_wiki_v0_from_json.csv",
                # 'edge_csv': ff"{npy_dir}/triple_edges_en_simple_wiki_v0_from_json.csv",
                'text_csv': f"{npy_dir}/text_nodes_en_simple_wiki_v0_from_json_with_emb.csv",
            }
        }
    }
    emb_list = ['node', 'text']  # Add 'edge' if needed and uncomment the related path lines
    for emb in emb_list:
        npy_path = keyword_to_paths[keyword]['npy'][emb]
        index_path = keyword_to_paths[keyword]['index'][emb]
        index_type = keyword_to_paths[keyword]['index_type'][emb]
        logging.info(f"Index {index_path}, Building...")
        # For cc-en the recommended training samples is 600_000_000, for the rest we can afford to training them using all data.
        build_faiss_from_npy(index_type, index_path, npy_path)        

    # # Test the index
    # for emb in emb_list:
    #     index_path = keyword_to_paths[keyword]['index'][emb]
    #     print(f"Index {index_path}, Testing...")
    #     test_and_search_faiss_index(index_path, keyword_to_paths[keyword]['csv'][emb])

if __name__ == "__main__":

    x = 1

    # keyword = "cc_en"  # Replace with your actual keyword
    # logging.basicConfig(
    #     filename=f'{keyword}_faiss_creation.log',  # Log file
    #     level=logging.INFO,       # Set the logging level
    #     format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
    # )

    # argparser = argparse.ArgumentParser(description="Train and write FAISS indexes for LKG construction.")
    # argparser.add_argument("--npy_dir", type=str, default="./import", help="Directory containing the .npy files.")
    # argparser.add_argument("--keyword", type=str, default=keyword, help="Keyword to select the dataset.")
    
    # args = argparser.parse_args()
    # keyword = args.keyword
    # npy_dir = args.npy_dir
    
    # train_and_write_indexes(keyword,npy_dir)
    # index_type = "IVF65536_HNSW32,Flat"
    index_type = "HNSW,Flat"

    output_directory = "/home/jbai/AutoSchemaKG/import/Dulce"
    filename_pattern = "Dulce"

    build_faiss_from_npy(
            index_type=index_type,
        index_path=f"{output_directory}/vector_index/triple_nodes_{filename_pattern}_from_json_with_emb_non_norm.index",
        npy_path=f"{output_directory}/vector_index/triple_nodes_{filename_pattern}_from_json_with_emb.npy",
    )

    build_faiss_from_npy(
        index_type=index_type,
        index_path=f"{output_directory}/vector_index/text_nodes_{filename_pattern}_from_json_with_emb_non_norm.index",
        npy_path=f"{output_directory}/vector_index/text_nodes_{filename_pattern}_from_json_with_emb.npy",
    )

    build_faiss_from_npy(
        index_type=index_type,
        index_path=f"{output_directory}/vector_index/triple_edges_{filename_pattern}_from_json_with_concept_with_emb_non_norm.index",
        npy_path=f"{output_directory}/vector_index/triple_edges_{filename_pattern}_from_json_with_concept_with_emb.npy",
    )