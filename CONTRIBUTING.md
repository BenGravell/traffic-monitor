# Contribution Guidelines

## File validation and unit tests

Run

```bash
pre-commit run --all-files
```

to run all pre-commit hooks, including file linting, mypy, and unit tests.

## Modifying packages

Add new direct dependencies to `requirements.in` then run

```bash
pip-compile requirements.in
```

to compile direct dependencies to `requirements.txt`.

Run

```bash
pip install -r requirements.txt
```

to actually install the required packages.
