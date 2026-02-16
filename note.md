```bash
cd benchbase/target/benchbase-postgres
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --create true --load true
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --execute true 

PYTHONPATH=src python src/run_gptuner.py postgres tpch 180 -seed=100
PYTHONPATH=src python src/run_gptuner.py src/exp_configs/tpcc/claude-sonnet4-overall.yml
PYTHONPATH=src python src/run_gptuner.py src/exp_configs/tpcc/gpt5.2-overall.yml
PYTHONPATH=src python src/run_gptuner_catune.py src/exp_configs/tpch/catune-gpt-5.2-seed100-pure.yml
```


## Token usage

- gpt-4o-mini: 681,068
- gpt-4o: 988,321 (containing 1 complete knowledge and 1/2 knowledge)

## YML file setup guidelines

- `folder`
    - `default`: make a new dir for the result with dir name `{yml_file_name}_{current_time}`
- `process`
    - `whole`: run knowledge collection and optimization
    - `knowledge`: run knowedge collection only
    - `optimization`: run optimization only. when set to `optimization`, the `folder` must be specified to a dir with tuning knowledge and the optimization results would be stored in the `folder`
        - if you want to rerun optimization with existing knowledge. Please use `knob`, `suggest_range_path`, `special_skill_path`. 
- `knob`
    - Specify the file path of the picked knobs, e.g., `./knowledge_collection/postgres/target_knobs.txt`, then `KnowledgeSelection` would be skipped. Default: None, it would run `KnowledgeSelection` to pick knobs.
- `suggest_range_path`
    - Specify the project folder path for their suggested range (`normal`), e.g., `./knowledge_collection/postgres/structured_knowledge/normal`, 

