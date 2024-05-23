# linkml-model-generator
Translate pydantic-based schemas from python modules to LinkML.

### Run

To run the program, use:

```
  python3 generate_linkml_from_aind.py --root_module_name="aind_data_schema.models" --output_file="aind.yml"
```

### Limitations

1. Supports only Pydantic models with maximum of one immediate parent that is a Pydantic model. 
