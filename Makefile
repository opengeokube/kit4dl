.PHONY: typehint
typehint:
	mypy --ignore-missing-imports src/
	pylint src/

.PHONY: test
test:
	pytest tests/

.PHONY: format
format:
	black -l 79 --preview .

.PHONY: docs
docs:
	pydocstyle -e --convention=numpy src

.PHONY: clean
clean:
	rm -rf dist
	find . -type f -name '*.pyc' | xargs rm -fr
	find . -type d -name "__pycache__" | xargs rm -fr
	find . -type d -name ".ipynb_checkpoints" | xargs rm -fr

prepublish: format typehint docs test
