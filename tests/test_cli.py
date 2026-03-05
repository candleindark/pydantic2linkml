from pathlib import Path

import pytest
from linkml_runtime.linkml_model import SchemaDefinition
from typer.testing import CliRunner

from pydantic2linkml.cli import app, main

runner = CliRunner()

_MOCK_SCHEMA = SchemaDefinition(id="https://example.com/test", name="test-schema")


def test_smoke_cli():
    result = runner.invoke(app, ["dandischema.models"])
    assert result.exit_code == 0


def test_cli_command_func():
    """Test calling the CLI command function directly"""
    main(["dandischema.models"])


class TestCliOverlay:
    @pytest.fixture(autouse=True)
    def mock_translate_defs(self, mocker):
        # autouse=True with a fixture defined inside a class automatically applies
        # it to every test in the class. Class scope would be more appropriate
        # since the mock is identical for all tests, but mocker is function-scoped
        # and pytest prohibits a fixture from depending on one of narrower scope.
        mocker.patch("pydantic2linkml.cli.translate_defs", return_value=_MOCK_SCHEMA)

    def test_valid_field(self, tmp_path: Path):
        overlay_file = tmp_path / "overlay.yaml"
        overlay_file.write_text("name: my-name\n")
        result = runner.invoke(app, ["dandischema.models", "-O", str(overlay_file)])
        assert result.exit_code == 0
        assert "name: my-name" in result.output

    def test_nonexistent_file(self, tmp_path: Path):
        result = runner.invoke(
            app,
            ["dandischema.models", "-O", str(tmp_path / "no-such-file.yaml")],
        )
        assert result.exit_code == 2
        assert "overlay file path is invalid" in result.output.lower()

    def test_non_mapping(self, tmp_path: Path):
        overlay_file = tmp_path / "overlay.yaml"
        overlay_file.write_text("- item1\n")
        result = runner.invoke(app, ["dandischema.models", "-O", str(overlay_file)])
        assert result.exit_code == 2
        assert "does not contain a" in result.output.lower()

    def test_unknown_key(self, tmp_path: Path):
        overlay_file = tmp_path / "overlay.yaml"
        overlay_file.write_text("not_a_field: some_value\n")
        result = runner.invoke(app, ["dandischema.models", "-O", str(overlay_file)])
        assert result.exit_code == 0
        assert "not_a_field" not in result.output
