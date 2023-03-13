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
		pytest test -k multi_device
else
		# default
		pytest test
endif

FORCE:
