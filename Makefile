.PHONY: dev-venv
dev-venv:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r dev-requirements.txt --upgrade --no-cache-dir

.PHONY: venv
venv: dev-venv
	venv/bin/pip install -r requirements.txt --upgrade --no-cache-dir

.PHONY: require_inside_venv
require_inside_venv:
	@if [[ -z "$$VIRTUAL_ENV" ]]; then echo "Activate the virtualenv first: \`. venv/bin/activate\`"; exit 1; fi

.PHONY: format
format:
	venv/bin/black --line-length 100 --py36 --exclude "build/|buck-out/|dist/|_build/|\.eggs/|\*\*/\.eggs|\.git/|\.hg/|\.mypy_cache/|\.nox/|\.tox/|venv|\.venv/|\*\*/\venv|\*\*/\.venv" .

.PHONY: lint
lint:
	venv/bin/flake8
