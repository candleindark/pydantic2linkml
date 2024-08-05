from typer.testing import CliRunner

from pydantic2linkml.cli import app

runner = CliRunner()


def test_smoke_cli():
    result = runner.invoke(app, ["dandischema.models"])
    assert result.exit_code == 0
