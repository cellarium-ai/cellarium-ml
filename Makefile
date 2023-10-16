.PHONY: install lint license format test FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall cellarium-ml

lint: FORCE
	ruff check .
	black --check .

license: FORCE
	python scripts/update_headers.py

format: license FORCE
	ruff check --fix .
	black .

typecheck: FORCE
	mypy cellarium test

test: FORCE
ifeq (${TEST_DEVICES}, 2)
	pytest -v -k multi_device
else ifeq (${TEST_DEVICES}, 1)
	# default
	pytest -v -n auto
endif

FORCE:
