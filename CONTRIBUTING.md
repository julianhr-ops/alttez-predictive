# Contributing

Use Python 3.13. Create a virtual environment, activate it, and install the dependencies.

```sh
python -m venv venv
python -m pip install -r requirements.txt
python -m pip install ruff==0.17.5
```

Run the tests and lint before submitting changes.

```sh
python -m unittest discover -s tests
python -m ruff check .
```

Tests must run offline. Use local fixtures and mock external services instead of making network calls.

Create a branch for each change instead of working on `main`. Use focused commits with messages such as `fix(model): handle missing results`. Open a pull request for review.
