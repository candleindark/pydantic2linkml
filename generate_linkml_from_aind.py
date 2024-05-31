from pathlib import Path

import pydantic
import enum
import inspect
import typer

from linkml_runtime.linkml_model import EnumDefinition
from linkml_runtime.utils.schema_builder import SchemaBuilder
from linkml_runtime.dumpers import yaml_dumper

from pydantic2linkml.tools import get_all_modules

app = typer.Typer()

# BICAN already has linkml here:
#   https://github.com/brain-bican/models/tree/main/linkml-schema
# Biolink also has linkml:
#   https://github.com/biolink/biolink-model/blob/master/src/biolink_model/schema/biolink_model.yaml
# openminds is JSON: https://github.com/openMetadataInitiative/openMINDS_core/tree/v4
# ATOM: https://bioportal.bioontology.org/ontologies/ATOM
# ATOM: https://github.com/SciCrunch/NIF-Ontology/blob/atlas/ttl/atom.ttl
# ATOM: https://www.nature.com/articles/s41597-023-02389-4
KNOWN_MODELS = {"dandi": "dandischema.models", "aind": "aind_data_schema.models"}


def populate_enum(sb: SchemaBuilder, enum_name: str, enum_object: type[enum.Enum]):
    """
    Populate a LinkML SchemaBuilder instance with a new enum derived from
    a pydantic Enum object.
    """
    try:
        sb.add_enum(
            EnumDefinition(
                name=enum_name,
                permissible_values=dict(
                    (attribute, getattr(enum_object, attribute).value)
                    for attribute in dir(enum_object)
                    if not attribute.startswith("__")
                    and isinstance(getattr(enum_object, attribute), enum.Enum)
                ),
            )
        )
    except ValueError as e:
        if "already exists" not in str(e):
            raise


def populate_basemodel(
    sb: SchemaBuilder, basemodel_name: str, basemodel_object: type[pydantic.BaseModel]
):
    sb.add_class(
        basemodel_name,
        slots=basemodel_object.__annotations__,
        is_a=basemodel_object.__mro__[1].__name__,
        class_uri=f"schema:{basemodel_name}",
        description=(
            basemodel_object.__doc__.strip()
            if basemodel_object.__doc__
            else "No description"
        ),
    )


def populate_schema_builder_from_module(sb: SchemaBuilder, module: str):
    for module in get_all_modules(root_module_name=module):
        for class_name, class_object in inspect.getmembers(module, inspect.isclass):
            if issubclass(class_object, enum.Enum):
                populate_enum(sb, class_name, class_object)
            elif issubclass(class_object, pydantic.BaseModel):
                populate_basemodel(sb, class_name, class_object)


@app.command()
def main(
    root_module_name: str = KNOWN_MODELS["aind"],
    output_file: Path = Path("generated_linkml_models/aind.yml"),
):
    sb = SchemaBuilder()
    populate_schema_builder_from_module(sb, module=root_module_name)
    yml = yaml_dumper.dumps(sb.schema)
    with output_file.open("w") as f:
        f.write(yml)
    print("Success!")


if __name__ == "__main__":
    typer.run(main)
