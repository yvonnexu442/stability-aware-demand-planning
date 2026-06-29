.PHONY: smoke experiment

smoke:
	PYTHONPATH=src python3 -m unittest discover -s tests

experiment:
	PYTHONPATH=src python3 -m experiments.run_experiment --dataset synthetic_demo
