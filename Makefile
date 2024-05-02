.PHONY: install lint license format test FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall cellarium-ml

lint: FORCE
	ruff check .
	ruff format --check .

docs: FORCE
	cd docs && make html

license: FORCE
	python scripts/update_headers.py

format: license FORCE
	ruff check --fix .
	ruff format .

typecheck: FORCE
	mypy cellarium tests

test: FORCE
ifeq (${TEST_DEVICES}, 2)
	pytest -v -k multi_device
else
	# default
	pytest -v
endif

FORCE:
