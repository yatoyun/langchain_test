
run:
	uv run python agent_book_7evaluation.py

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

fix:
	uv run ruff check --fix .

sync:
	uv sync