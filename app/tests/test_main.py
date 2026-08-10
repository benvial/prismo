"""Tests for the main application module."""

from importlib import import_module

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_import() -> None:
    """Placeholder test that ensures package import works as expected."""
    import_module("prismo")


def test_cli_help() -> None:
    """Test the CLI help command."""
    main_module = import_module("prismo.main")
    result = runner.invoke(main_module.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.stdout


def test_cli_run_help() -> None:
    """Test the CLI run subcommand help."""
    main_module = import_module("prismo.main")
    result = runner.invoke(main_module.app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "Run the PRISMO" in result.stdout


@pytest.mark.slow
def test_cli_run_synthetic() -> None:
    """Test `prismo run` on a synthetic mesh (no gmsh)."""
    main_module = import_module("prismo.main")
    result = runner.invoke(
        main_module.app,
        ["--max-iter", "3", "--no-jit"],
    )
    assert result.exit_code == 0, result.output
    assert "Done" in result.stdout
