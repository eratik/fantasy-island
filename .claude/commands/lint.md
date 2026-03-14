Lint and format the Python backend code with ruff. Fix any issues found automatically where possible.

```bash
cd RunpodBackend && uv run ruff check --fix . && uv run ruff format .
```

Report any remaining issues that couldn't be auto-fixed.
