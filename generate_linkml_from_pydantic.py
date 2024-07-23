#!/usr/bin/env python3

from pathlib import Path

import pydantic
import enum
import inspect

from linkml_runtime.linkml_model import EnumDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from linkml_runtime.dumpers import yaml_dumper

from pydantic2linkml.tools import get_all_modules
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

def get_schema_for_module(module: str) -> ...:
    models, enums = [], []
    for module in get_all_modules(root_module_name=module):
        for class_name, class_object in inspect.getmembers(module, inspect.isclass):
            if issubclass(class_object, enum.Enum):
                enums.append(class_object)
            elif issubclass(class_object, pydantic.BaseModel):
                models.append(class_object)
    generator = LinkmlGenerator(
        name=module,
        enums=enums,
        models=models,
    )
    return generator.generate()


def main(
    root_module_name: str,
    output_file: Path = None,
):
    schema = get_schema_for_module(module=root_module_name)
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
