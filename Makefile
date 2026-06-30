.PHONY: smoke

smoke:
	PYTHONPATH=src python3 -m unittest discover -s tests
