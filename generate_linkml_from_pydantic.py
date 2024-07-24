#!/usr/bin/env python3

from pathlib import Path

import pydantic
import enum
import inspect

from linkml_runtime.linkml_model import EnumDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from linkml_runtime.dumpers import yaml_dumper

from pydantic2linkml.tools import get_all_modules, fetch_defs
from pydantic2linkml.gen_linkml import LinkmlGenerator


# BICAN already has linkml here:
#   https://github.com/brain-bican/models/tree/main/linkml-schema
# Biolink also has linkml:
#   https://github.com/biolink/biolink-model/blob/master/src/biolink_model/schema/biolink_model.yaml
# openminds is JSON: https://github.com/openMetadataInitiative/openMINDS_core/tree/v4
# ATOM: https://bioportal.bioontology.org/ontologies/ATOM
# ATOM: https://github.com/SciCrunch/NIF-Ontology/blob/atlas/ttl/atom.ttl
# ATOM: https://www.nature.com/articles/s41597-023-02389-4
# no longer used: KNOWN_MODELS = {"dandi": "dandischema.models", "aind": "aind_data_schema.models"}


def main(
    module_names: list[str],
    output_file: Path = None,
):
    # TODO: RF prints to log messages to stderr
    modules = get_all_modules(module_names)
    print(
        f"Considering {len(modules)} modules for provided {len(module_names)} modules: "
        f"{module_names}"
    )
    models, enums = fetch_defs(modules)
    print(f"Fetched {len(models)} models and {len(enums)} enums")
    generator = LinkmlGenerator(models=models, enums=enums)
    print("Generating schema")
    schema = generator.generate()
    print("Dumping schema")
    yml = yaml_dumper.dumps(schema)
    if not output_file:
        print(yml)
    else:
        with output_file.open("w") as f:
            f.write(yml)
    print("Success!")


if __name__ == "__main__":
    import typer

    app = typer.Typer()
    typer.run(app.command()(main))
