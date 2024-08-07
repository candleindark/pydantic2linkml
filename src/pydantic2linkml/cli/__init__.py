from pathlib import Path

import typer
from linkml_runtime.dumpers import yaml_dumper

from pydantic2linkml.gen_linkml import LinkmlGenerator
from pydantic2linkml.tools import fetch_defs, get_all_modules

app = typer.Typer()


@app.command()
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
