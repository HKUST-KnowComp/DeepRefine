python benchmark/autograph/benchmarking_text_refiner.py \
    --refine_max_subset_size 1000 \
    --refine_subset_target_coverage 0.8 \
    --refine_subset_topk 10 \
    --refine_subset_1hop_sample_size 500 \
    --refine_subset_workers 16 \
    --refine \
    --kg_type graphify