# quant_research
## Commands
### Turn project into zip
uv run python scripts/bundle_project.py --output ~/Downloads/project_context.zip

### Verification
uv run ruff format
uv run ruff check .
uv run ruff format --check .
uv run mypy src/quant_research
uv run pytest