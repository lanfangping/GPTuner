```bash
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --create true --load true
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --execute true 

PYTHONPATH=src python src/run_gptuner.py postgres tpch 180 -seed=100
PYTHONPATH=src python src/run_gptuner.py src/exp_configs/test_config.yml
```