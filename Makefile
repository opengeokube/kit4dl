.PHONY: typehint
typehint:
	mypy --ignore-missing-imports .
	pylint kit4dl

.PHONY: test
test:
	pytest tests/

.PHONY: format
format:
	# isort .git 
	black .

.PHONY: docs
docs:
	pydocstyle -e --convention=numpy kit4dl

prepublish: format typehint docs test
