# Notes

```bash
PYTHONPATH=src python src/run_gptuner.py postgres tpcc 180 -seed=100 -enhanced=True
PYTHONPATH=src python src/knowledge_handler/knowledge_preparation.py

PYTHONPATH=src python src/run_gptuner.py --config src/exp_configs/postgres/sf20_t10/noknowledge_coarseD_fineD.yml

PYTHONPATH=src python src/run.py
```