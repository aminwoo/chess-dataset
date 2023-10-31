lint:
	python -m pylint --version
	pylint ./src

black:
	black ./src