from collections.abc import Iterable

import pytest

TRANSLATOR_PACKAGE = "pydantic2linkml"


def has_exactly_one_truthy(iterable: Iterable) -> bool:
    """
    Determine if exactly one element in an iterable is truthy.

    :param iterable: The iterable
    """
    return sum(map(bool, iterable)) == 1


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

        slot_gen = SlotGenerator(field_schema)

        assert slot_gen._slot.name == "x"
        assert slot_gen._field_schema == field_schema

        # Test the _schema_type_to_method mapping at selective keys
        assert slot_gen._schema_type_to_method["any"] == slot_gen._any_schema
        assert slot_gen._schema_type_to_method["bool"] == slot_gen._bool_schema
        assert slot_gen._schema_type_to_method["model"] == slot_gen._model_schema

        assert not slot_gen._used

    def test_any_schema(self):
        from typing import Any

        from pydantic import BaseModel

        from pydantic2linkml.gen_linkml import SlotGenerator
        from pydantic2linkml.tools import get_field_schema

        class Foo(BaseModel):
            x: Any

        field_schema = get_field_schema(Foo, "x")
        slot = SlotGenerator(field_schema).generate()
        assert slot.range == "Any"

    def test_none_schema(self):
        from pydantic import BaseModel

        from pydantic2linkml.gen_linkml import SlotGenerator
        from pydantic2linkml.tools import get_field_schema

        class Foo(BaseModel):
            x: None

        field_schema = get_field_schema(Foo, "x")
        slot = SlotGenerator(field_schema).generate()
        assert len(slot.notes) == 1
        assert (
            slot.notes[0] == f"{TRANSLATOR_PACKAGE}: LinkML does not have null values. "
            f"(For details, see https://github.com/orgs/linkml/discussions/1975)."
        )

    def test_bool_schema(self):
        from pydantic import BaseModel

        from pydantic2linkml.gen_linkml import SlotGenerator
        from pydantic2linkml.tools import get_field_schema

        class Foo(BaseModel):
            x: bool

        field_schema = get_field_schema(Foo, "x")
        slot = SlotGenerator(field_schema).generate()
        assert slot.range == "boolean"

    @pytest.mark.parametrize(
        "multiple_of, le, lt, ge, gt, expected_max, expected_min",
        [
            (2, 100, 101, 0, -1, 100, 0),
            (2, 100, 200, -10, -100, 100, -10),
            (2, 200, 100, -100, -10, 99, -9),
            (None, 200, 100, -100, -10, 99, -9),
            (2, 200, None, -100, None, 200, -100),
            (2, None, 100, None, -10, 99, -9),
            (None, None, None, None, None, None, None),
        ],
    )
    def test_int_schema(self, multiple_of, le, lt, ge, gt, expected_max, expected_min):
        from pydantic import BaseModel, Field

        from pydantic2linkml.gen_linkml import SlotGenerator
        from pydantic2linkml.tools import get_field_schema

        class Foo(BaseModel):
            x: int = Field(..., multiple_of=multiple_of, le=le, ge=ge, lt=lt, gt=gt)

        field_schema = get_field_schema(Foo, "x")
        slot = SlotGenerator(field_schema).generate()

        assert slot.range == "integer"

        if multiple_of is not None:
            assert len(slot.notes) == 1
            assert f"a multiple of {multiple_of}" in slot.notes[0]
        else:
            assert len(slot.notes) == 0

        assert slot.maximum_value == expected_max
        assert slot.minimum_value == expected_min

    # noinspection DuplicatedCode
    @pytest.mark.parametrize("allow_inf_nan", [True, False, None])
    @pytest.mark.parametrize("multiple_of", [2, 42, None])
    @pytest.mark.parametrize("le", [100, -11, None])
    @pytest.mark.parametrize("ge", [10, -42, None])
    @pytest.mark.parametrize("lt", [10, -11, None])
    @pytest.mark.parametrize("gt", [100, -120, None])
    def test_float_schema(self, allow_inf_nan, multiple_of, le, ge, lt, gt):
        from pydantic import BaseModel, Field

        from pydantic2linkml.gen_linkml import SlotGenerator
        from pydantic2linkml.tools import get_field_schema

        class Foo(BaseModel):
            x: float = Field(
                ...,
                allow_inf_nan=allow_inf_nan,
                multiple_of=multiple_of,
                le=le,
                ge=ge,
                lt=lt,
                gt=gt,
            )

        field_schema = get_field_schema(Foo, "x")
        slot = SlotGenerator(field_schema).generate()

        assert slot.range == "float"

        msg_in_note_lst = [
            "LinkML does not have support for `'+inf'`, `'-inf'`, and `'NaN'`" in note
            for note in slot.notes
        ]
        if allow_inf_nan is None or allow_inf_nan:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [f"multiple of {multiple_of}" in note for note in slot.notes]
        if multiple_of is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        assert slot.maximum_value == le
        assert slot.minimum_value == ge

        msg_in_note_lst = [f"less than {lt}" in note for note in slot.notes]
        if lt is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [f"greater than {gt}" in note for note in slot.notes]
        if gt is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

    # noinspection DuplicatedCode
    @pytest.mark.parametrize(
        "allow_inf_nan, max_digits, decimal_places",
        [(True, None, None), (False, 3, 42), (None, 2, 4), (False, None, 2)],
    )
    @pytest.mark.parametrize("multiple_of", [2, 42, None])
    @pytest.mark.parametrize("le", [100, -11, None])
    @pytest.mark.parametrize("ge", [10, -42, None])
    @pytest.mark.parametrize("lt", [10, -11, None])
    @pytest.mark.parametrize("gt", [100, -120, None])
    def test_decimal_schema(
        self, allow_inf_nan, max_digits, decimal_places, multiple_of, le, ge, lt, gt
    ):
        from decimal import Decimal
        from pydantic import BaseModel, Field

        from pydantic2linkml.gen_linkml import SlotGenerator
        from pydantic2linkml.tools import get_field_schema

        class Foo(BaseModel):
            x: Decimal = Field(
                ...,
                allow_inf_nan=allow_inf_nan,
                multiple_of=multiple_of,
                le=le,
                ge=ge,
                lt=lt,
                gt=gt,
                max_digits=max_digits,
                decimal_places=decimal_places,
            )

        field_schema = get_field_schema(Foo, "x")
        slot = SlotGenerator(field_schema).generate()

        assert slot.range == "decimal"

        msg_in_note_lst = [
            "LinkML does not have support for `'+inf'`, `'-inf'`, and `'NaN'`" in note
            for note in slot.notes
        ]
        if allow_inf_nan is not None and allow_inf_nan:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [
            f"max number of {max_digits} digits" in note for note in slot.notes
        ]
        if max_digits is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [
            f"max number of {decimal_places} decimal places" in note
            for note in slot.notes
        ]
        if decimal_places is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [f"multiple of {multiple_of}" in note for note in slot.notes]
        if multiple_of is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        assert slot.maximum_value == le
        assert slot.minimum_value == ge

        msg_in_note_lst = [f"less than {lt}" in note for note in slot.notes]
        if lt is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [f"greater than {gt}" in note for note in slot.notes]
        if gt is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

    # noinspection DuplicatedCode
    @pytest.mark.parametrize(
        "pattern, max_length, min_length, output_pattern",
        [
            (None, 10, 4, r"^(?=.{4,10}$)"),
            (None, None, 4, r"^(?=.{4,}$)"),
            (None, 10, None, r"^(?=.{,10}$)"),
            (r"^[a-zA-Z0-9]+", 3, 2, r"^(?=.{2,3}$)[a-zA-Z0-9]+"),
            (r"^[a-zA-Z0-9]+", None, 2, r"^(?=.{2,}$)[a-zA-Z0-9]+"),
            (r"^[a-zA-Z0-9]+", 3, None, r"^(?=.{,3}$)[a-zA-Z0-9]+"),
            (ptrn := r"^[a-zA-Z0-9]+", None, None, ptrn),
            (r".*", 10, 4, r"^(?=.{4,10}$).*"),
            (r".*", None, 4, r"^(?=.{4,}$).*"),
            (r".*", 10, None, r"^(?=.{,10}$).*"),
            (ptrn := r".*", None, None, ptrn),
            (None, None, None, None),
        ],
    )
    @pytest.mark.parametrize("strip_whitespace", [True, False, None])
    @pytest.mark.parametrize("to_lower", [True, False, None])
    @pytest.mark.parametrize("to_upper", [True, False, None])
    def test_str_schema(
        self,
        pattern,
        max_length,
        min_length,
        strip_whitespace,
        to_lower,
        to_upper,
        output_pattern,
    ):
        from pydantic import BaseModel, StringConstraints
        from typing_extensions import Annotated

        from pydantic2linkml.gen_linkml import SlotGenerator
        from pydantic2linkml.tools import get_field_schema

        class Foo(BaseModel):
            # noinspection PyTypeHints
            x: Annotated[
                str,
                StringConstraints(
                    pattern=pattern,
                    max_length=max_length,
                    min_length=min_length,
                    strip_whitespace=strip_whitespace,
                    to_lower=to_lower,
                    to_upper=to_upper,
                ),
            ]

        field_schema = get_field_schema(Foo, "x")
        slot = SlotGenerator(field_schema).generate()

        assert slot.range == "string"

        assert slot.pattern == output_pattern

        msg_in_note_lst = [
            f"The max length constraint of {max_length} is incorporated" in n
            for n in slot.notes
        ]
        if max_length is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [
            f"The min length constraint of {min_length} is incorporated" in n
            for n in slot.notes
        ]
        if min_length is not None:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [
            "stripping leading and trailing whitespace in LinkML" in n
            for n in slot.notes
        ]
        if strip_whitespace is not None and strip_whitespace:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [
            "Unable to express the option of converting the string to lowercase" in n
            for n in slot.notes
        ]
        if to_lower is not None and to_lower:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)

        msg_in_note_lst = [
            "Unable to express the option of converting the string to uppercase" in n
            for n in slot.notes
        ]
        if to_upper is not None and to_upper:
            assert has_exactly_one_truthy(msg_in_note_lst)
        else:
            assert not any(msg_in_note_lst)
