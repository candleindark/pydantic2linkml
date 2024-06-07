import pytest


class TestGenLinkml:
    @pytest.mark.parametrize(
        "root_module_name",
        [
            "dandischema.models",
            "aind_data_schema.components.coordinates",
            # Naming conflict at this one
            # TODO: Re-enable this one once handling of the naming conflict is devised
            # "aind_data_schema.components.devices",
            "aind_data_schema.components.reagent",
            "aind_data_schema.components.stimulus",
            "aind_data_schema.components.tile",
            "aind_data_schema.core.acquisition",
            "aind_data_schema.core.data_description",
            "aind_data_schema.core.instrument",
            "aind_data_schema.core.metadata",
            "aind_data_schema.core.procedures",
            "aind_data_schema.core.processing",
            "aind_data_schema.core.rig",
            "aind_data_schema.core.session",
            "aind_data_schema.core.subject",
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

    def test_establish_supporting_defs(self):
        """
        Verify the setting of supporting definitions in the schema associated with
            a GenLinkml object.

        The setting of the supporting definitions is done by the
            `_establish_supporting_defs()` method.
        """
        from pydantic2linkml.gen_linkml import LinkmlGenerator

        gen = LinkmlGenerator()
        schema = gen._sb.schema

        assert "Any" in schema.classes
        assert schema.classes["Any"].name == "Any"
        assert schema.classes["Any"].description == "Any object"
        assert schema.classes["Any"].class_uri == "linkml:Any"


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
