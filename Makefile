.PHONY: install lint format test FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall scvi-distributed

lint: FORCE
	flake8
	black --check .
	isort --check .

format: FORCE
	black .
	isort .

test: lint FORCE
ifeq (${TEST_DEVICES}, 2)
	pytest -v -k multi_device
else (${TEST_DEVICES}, 1)
	# default
	pytest -v -n auto
endif

FORCE:
