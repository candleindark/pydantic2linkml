import pytest


class TestGenLinkml:
    @pytest.mark.parametrize(
        "root_module_name",
        [
            "dandischema.models",
            # "aind_data_schema.models" todo: to be added later
            # currently, it causes name collision due to
            # multiple classes from different modules having the
            # same name
        ],
    )
    def test_instantiation_with_definitions_in_module(self, root_module_name):
        """
        Test instantiation of a `GenLinkml` object with Pydantic models and enums from
        a module and its supporting modules.

        :param root_module_name: The name of the module, the root module, importing
            of which necessitates the importing of its supporting modules.
        """
        from pydantic2linkml.tools import get_all_modules, fetch_defs
        from pydantic2linkml.gen_linkml import LinkmlGenerator

        models, enums = fetch_defs(get_all_modules(root_module_name))
        LinkmlGenerator(models=models, enums=enums)


class TestSlotGenerator:
    def test_instantiation(self):
        from pydantic import BaseModel
        from pydantic2linkml.gen_linkml import SlotGenerator
        from pydantic2linkml.tools import get_field_schema

        class Foo(BaseModel):
            x: int

        field_schema = get_field_schema(Foo, "x")

        slot_gen = SlotGenerator("Eks", field_schema)

        assert slot_gen._slot.name == "Eks"
        assert slot_gen._field_schema == field_schema

        # Test the _schema_type_to_method mapping at selective keys
        assert slot_gen._schema_type_to_method["any"] == slot_gen._any_schema
        assert slot_gen._schema_type_to_method["bool"] == slot_gen._bool_schema
        assert slot_gen._schema_type_to_method["model"] == slot_gen._model_schema

        assert not slot_gen._used
