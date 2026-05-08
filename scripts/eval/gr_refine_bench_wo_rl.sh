python benchmark/autograph/benchmarking_graph_refiner.py \
    --refine_max_subset_size 1000 \
    --refine_subset_target_coverage 1.0 \
    --refine_subset_topk 10 \
    --refine_subset_1hop_sample_size 100 \
    --refine_subset_workers 16 \
    --refine \
    --kg_type graphify