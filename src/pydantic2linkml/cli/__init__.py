import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from linkml_runtime.dumpers import yaml_dumper

from pydantic2linkml.cli.tools import LogLevel
from pydantic2linkml.gen_linkml import translate_defs

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    module_names: list[str],
    overlay_file: Annotated[
        Optional[Path],
        typer.Option(
            "--overlay-file",
            "-O",
            help="An overlay file, specifying a (partial) schema, to be applied on top "
            "of the generated schema.",
        ),
    ] = None,
    output_file: Annotated[Optional[Path], typer.Option("--output-file", "-o")] = None,
    log_level: Annotated[
        LogLevel, typer.Option("--log-level", "-l")
    ] = LogLevel.WARNING,
):
    # Set log level of the CLI
    logging.basicConfig(level=getattr(logging, log_level))

    schema = translate_defs(module_names, overlay_file)
    logger.info("Dumping schema")
    yml = yaml_dumper.dumps(schema)
    if not output_file:
        print(yml)  # noqa: T201
    else:
        with output_file.open("w") as f:
            f.write(yml)
    logger.info("Success!")
