.PHONY: install ui test lint serve dev docker

install:
	pip install -e ".[dev]"

ui:
	cd frontend && npm ci && npm run build

test:
	pytest

lint:
	ruff check catforge tests
	cd frontend && npx tsc -b

serve:
	catforge serve --port 8000

dev:
	@echo "run 'catforge serve' and 'cd frontend && npm run dev' in two terminals"

docker:
	docker build -t catforge .
