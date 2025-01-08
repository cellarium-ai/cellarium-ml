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
	pytest -v -k multi_device --ignore=tests/dataloader --ignore=tests/test_mup.py
else
	# default
	pytest -v --ignore=tests/dataloader
endif

test-dataloader: FORCE
ifeq (${TEST_DEVICES}, 2)
	pytest -v -k multi_device tests/dataloader
else
	# default
	pytest -v tests/dataloader
endif

test-examples: FORCE
	cellarium-ml onepass_mean_var_std fit --config examples/cli_workflow/onepass_train_config.yaml
	cellarium-ml incremental_pca fit --config examples/cli_workflow/ipca_train_config.yaml
	cellarium-ml logistic_regression fit --config examples/cli_workflow/lr_train_config.yaml
	cellarium-ml logistic_regression fit --config examples/cli_workflow/lr_resume_train_config.yaml

FORCE:
