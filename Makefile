.PHONY: install lint license format test FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall scvi-distributed

lint: FORCE
	flake8
	black --check .
	isort --check .

license: FORCE
	python scripts/update_headers.py

format: license FORCE
	black .
	isort .

typecheck: FORCE
	mypy scvid test examples

test: FORCE
ifeq (${TEST_DEVICES}, 2)
	pytest -v -k multi_device
else ifeq (${TEST_DEVICES}, 1)
	# default
	pytest -v -n auto
endif

FORCE:
