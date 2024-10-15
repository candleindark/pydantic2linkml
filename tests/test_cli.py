from typer.testing import CliRunner

from pydantic2linkml.cli import app, main

runner = CliRunner()


def test_smoke_cli():
    result = runner.invoke(app, ["dandischema.models"])
    assert result.exit_code == 0


def test_cli_command_func():
    """Test calling the CLI command function directly"""
    main(["dandischema.models"])
