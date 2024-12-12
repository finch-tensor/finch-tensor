# finch-tensor-python

This is the beginnings of a sparse tensor library for Python, backed by the
[Finch.jl](https://github.com/finch-tensor/Finch.jl) tensor compiler.

## Source

The source code for `finch-tensor` is available on GitHub at [https://github.com/finch-tensor/finch-tensor-python](https://github.com/FinchTensor/finch-tensor-python)

## Installation

`finch-tensor` is available on PyPi, and can be installed with pip:
```bash
pip install finch-tensor
```

## Contributing

### Packaging

Finch uses [poetry](https://python-poetry.org/) for packaging.

To install for development, clone the repository and run:
```bash
poetry install --with test
```
to install the current project and dev dependencies.

### Publishing

The "Publish" GitHub Action is a manual workflow for publishing Python packages to PyPI using Poetry. It handles the version management based on the `pyproject.toml` file and automates tagging and creating GitHub releases.

#### Version Update

Before initiating the "Publish" action, update the package's version number in `pyproject.toml`. Follow semantic versioning guidelines for this update.

#### Triggering the Action

The action is triggered manually. Once the version in `pyproject.toml` is updated, manually start the "Publish" action from the GitHub repository's Actions tab.

#### Process and Outcomes

On successful execution, the action publishes the package to PyPI and tags the release in the GitHub repository. If the version number is not updated, the action fails to publish to PyPI, and no tagging or release is done. In case of failure, correct the version number and rerun the action.

#### Best Practices

- Ensure the version number in `pyproject.toml` is updated before triggering the action.
- Regularly check action logs for successful completion or to identify issues.

### Pre-commit hooks

To add pre-commit hooks, run:
```bash
poetry run pre-commit install
```

### Testing

Finch uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
poetry run pytest
```
