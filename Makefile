.PHONY: typehint
typehint:
	mypy --ignore-missing-imports .
	pylint mlkit

.PHONY: test
test:
	pytest tests/

.PHONY: format
format:
	isort --profile black .
	black -l 79 --preview .

.PHONY: docs
docs:
	pydocstyle -e --convention=numpy mlkit

.PHONY: clean
clean:
	rm -rf build
	rm -rf dist
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type f -name '*.pyc' | xargs rm -fr
	find . -type d -name "__pycache__" | xargs rm -fr
	find . -type d -name ".ipynb_checkpoints" | xargs rm -fr

prepublish: clean format typehint docs test clean
