import logging
from pathlib import Path
from typing import Optional

import typer
from linkml_runtime.dumpers import yaml_dumper

from pydantic2linkml.gen_linkml import LinkmlGenerator
from pydantic2linkml.tools import fetch_defs, get_all_modules

from .tools import LogLevel

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    module_names: list[str],
    output_file: Optional[Path] = None,
    log_level: LogLevel = LogLevel.WARNING,
):
    # Set log level of the CLI
    logging.basicConfig(level=getattr(logging, log_level))

    modules = get_all_modules(module_names)
    logger.info(
        "Considering %d modules for provided %d modules: %s",
        len(modules),
        len(module_names),
        module_names,
    )
    models, enums = fetch_defs(modules)
    logger.info("Fetched %d models and %d enums", len(models), len(enums))
    generator = LinkmlGenerator(models=models, enums=enums)
    logger.info("Generating schema")
    schema = generator.generate()
    logger.info("Dumping schema")
    yml = yaml_dumper.dumps(schema)
    if not output_file:
        print(yml)  # noqa: T201
    else:
        with output_file.open("w") as f:
            f.write(yml)
    logger.info("Success!")
