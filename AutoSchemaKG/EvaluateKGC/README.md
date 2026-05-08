## Evaluation for KG Quality

### Schema Quality
To evaluate the schema accuracy of our schema induction method, we conducted experiments across three datasets `FB15kET`, `YAGO43kET` and `wikiHow`, which are the dataset names you can change in our codes under `./evaluation_kg/schema_quality`.

In detail, config your file paths, dataset name, model name, etc. And run the command:
```shell
cd ./evaluation_kg/schema_quality
python conceptualization_contextulized_api.py
```

Then evaluate the result file with running the command:
```shell
python eval.py
```
And the configurations are similar to those in `conceptualization_contextulized_api.py`.

### Triple Accuracy
To evaluate the accuracy of the triples in our KG, we use a rigorous counting-based evaluation method with DeepSeek-V3.

Replace the input file path and output file path and run the command:
```shell
cd ./evaluation_kg/triple_accuracy
python triple_acc.py
```

### Information Preservation
To evaluate the abilities of entity-level triples and event-level triples of our constructed KGs in preserving information from original passages, we tested with multi-choice questions(MCQs).

First, run the command to generate MCQs with original passages:
```shell
cd ./evaluation_kg/info_preservation
python question_generation.py
```

Then, test them with zero context, full context, entity-level triples, and event-level triples by running the command:
```shell
python answer_generation.py
```
