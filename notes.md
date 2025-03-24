# Notes

```bash
PYTHONPATH=src python src/run_gptuner.py postgres tpcc 180 -seed=100 -kw=1
# kw=1 means collecting knowledge; kw=0 means using collected knowledge
PYTHONPATH=src python src/run_gptuner.py postgres tpch 180 -seed=100

PYTHONPATH=src python src/knowledge_handler/knowledge_preparation.py
```