import pytest
from typer.testing import CliRunner
import typer

from generate_linkml_from_pydantic import main

runner = CliRunner()


def test_smoke_cli():
    # Mimic the app creation in generate_linkml_from_pydantic
    app = typer.Typer()
    app.command()(main)

    result = runner.invoke(app, ["dandischema.models"])
    assert result.exit_code == 0
