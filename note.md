```bash
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --create true --load true
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --execute true 

PYTHONPATH=src python src/run_gptuner.py postgres tpch 180 -seed=100
PYTHONPATH=src python src/run_gptuner.py src/exp_configs/gpt4-4o-mini-overall.yml
```


## Token usage

- gpt-4o-mini: 681,068
- gpt-4o: 988,321 (containing 1 complete knowledge and 1/2 knowledge)