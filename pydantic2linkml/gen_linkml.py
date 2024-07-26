import re
from enum import Enum
from typing import Optional, Callable, Any, Union
from collections.abc import Iterable
from collections import defaultdict
from itertools import chain
from warnings import warn
from operator import itemgetter
from datetime import date

from pydantic import BaseModel

# noinspection PyProtectedMember
from pydantic._internal import _typing_extra

# noinspection PyProtectedMember
from pydantic._internal._core_utils import CoreSchemaOrField

from pydantic.json_schema import CoreSchemaOrFieldType
from pydantic_core import core_schema
from linkml_runtime.utils.schema_builder import SchemaBuilder
from linkml_runtime.linkml_model import (
    SchemaDefinition,
    ClassDefinition,
    EnumDefinition,
    PermissibleValue,
    SlotDefinition,
)
from linkml_runtime.linkml_model.meta import AnonymousSlotExpression

from .exceptions import UserError
from .tools import (
    ensure_unique_names,
    normalize_whitespace,
    get_locally_defined_fields,
    LocallyDefinedFields,
    FieldSchema,
    resolve_ref_schema,
    bucketize,
    get_uuid_regex,
)

# The LinkML Any type
# For more info, see https://linkml.io/linkml/schemas/advanced.html#linkml-any-type
any_class_def = ClassDefinition(
    name="Any", description="Any object", class_uri="linkml:Any"
)


class LinkmlGenerator:
    """
    Instances of this class are single-use LinkML generators.

    Note:
        Each instance of this class should only be used once to generate
            a LinkML schema.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        id_: Optional[str] = None,
        models: Optional[Iterable[type[BaseModel]]] = None,
        enums: Optional[Iterable[type[Enum]]] = None,
    ):
        """
        :param name: The name of the LinkML schema to be generated
        :param id_: The ID of the LinkML schema to be generated
        :param models: An iterable of Pydantic models to be converted to LinkML classes
            in the generated schema
        :param enums: An iterable of Enums to be converted to LinkML enums in
            the generated schema

        raises NameCollisionError: If there are classes with the same name in the
            combined collection of `models` and `enums`
        """
        models: Iterable[type[BaseModel]] = models if models is not None else set()
        enums: Iterable[type[Enum]] = enums if enums is not None else set()

        ensure_unique_names(*models, *enums)

        # Map of models to their locally defined fields
        self._m_f_map: dict[type[BaseModel], LocallyDefinedFields] = {
            m: get_locally_defined_fields(m) for m in models
        }

        self._enums = enums
        self._sb = SchemaBuilder(name, id_)

        # This changes to True after this generator generates a schema
        # (for preventing issues caused by accidental re-use
        # of this generator). See class docstring for more info.
        self._used = False

    def generate(self) -> SchemaDefinition:
        """
        Generate a LinkML schema from the models and enums provided to this generator.

        :return: The generated LinkML schema
        """
        if self._used:
            raise UserError(
                f"This {type(self).__name__} instance has already been used to generate"
                " a LinkML schema. You must create a new instance of "
                f"{type(self).__name__} to generate another schema."
            )
        else:
            self._used = True

        self._sb.add_defaults()
        self._establish_supporting_defs()

        self._add_enums()  # Add enums to the schema
        self._add_slots()  # Add slots to the schema
        self._add_classes()  # Add classes to the schema

        return self._sb.schema

    def _add_enums(self):
        """
        Add LinkML enum representations of the enums in `self._enums` to the schema
        """
        for enum_ in self._enums:
            # All permissible values in the enum in string form
            enum_value_strs = [str(member.value) for member in enum_]

            self._sb.add_enum(
                EnumDefinition(
                    name=enum_.__name__,
                    description=(
                        normalize_whitespace(enum_.__doc__)
                        if enum_.__doc__ is not None
                        else None
                    ),
                    permissible_values=[
                        PermissibleValue(text=value_str, meaning=value_str)
                        for value_str in enum_value_strs
                    ],
                )
            )

    def _add_slots(self):
        """
        Add the slots construed from the fields in `self._m_f_map` to the schema
        """
        # Extract all the newly defined fields from across all models
        new_fields: Iterable[tuple[str, FieldSchema]] = chain.from_iterable(
            v.new.items() for v in self._m_f_map.values()
        )

        buckets: defaultdict[str, list[FieldSchema]] = bucketize(
            new_fields, key_func=itemgetter(0), value_func=itemgetter(1)
        )

        warnings_msg: Optional[str] = None
        for f_name, schema_lst in buckets.items():
            if len(schema_lst) > 1:
                # Construct the list of classes that define the fields corresponding
                # to the field schemas in `schema_lst`
                cls_lst = []
                for s in schema_lst:
                    # Resolve the context schema into a model schema
                    model_schema = resolve_ref_schema(s.context, s.context)

                    assert model_schema["type"] == "model"

                    cls_lst.append(model_schema["cls"])

                new_warnings_msg = (
                    f"Field name collision @ {f_name} from {cls_lst!r}, "
                    f"{f_name} field definition from {cls_lst[0]!r} is used to specify "
                    f"slot {f_name}"
                )
                warnings_msg = (
                    new_warnings_msg
                    if warnings_msg is None
                    else f"{warnings_msg}; {new_warnings_msg}"
                )

        if warnings_msg is not None:
            warn(warnings_msg)

        # Add the slots to the schema
        for schema_lst in buckets.values():
            # Use the first schema in `schema_lst` to generate the slot
            slot = SlotGenerator(schema_lst[0]).generate()

            # Add the slot to the schema
            self._sb.add_slot(slot)

    def _add_classes(self):
        """
        Add the classes construed from the models in `self._m_f_map` to the schema
        """
        raise NotImplementedError("Method not yet implemented")
        # todo: Make sure to provide slot usage in the individual classes if needed

    def _establish_supporting_defs(self):
        """
        Establish the supporting definitions in the schema
        """
        # Add an `linkml:Any` class
        self._sb.add_class(any_class_def)


class SlotGenerator:
    """
    Instances of this class are single-use slot generators.

    Note:
        Each instance of this class should only be used once to generate
            a LinkML slot schema.
    """

    def __init__(self, field_schema: FieldSchema):
        """
        :param field_schema: The `FieldSchema` object specifying the Pydantic core
            schema of the corresponding field with context
        """
        self._slot: SlotDefinition = SlotDefinition(name=field_schema.field_name)
        self._field_schema: FieldSchema = field_schema
        self._schema_type_to_method = self._build_schema_type_to_method()

        # This changes to True after this generator generates a slot schema
        # (for preventing issues caused by accidental re-use
        # of this generator). See class docstring for more info.
        self._used: bool = False

    def _build_schema_type_to_method(
        self,
    ) -> dict[CoreSchemaOrFieldType, Callable[[CoreSchemaOrField], None]]:
        """Builds a dictionary mapping schema and field types to methods for
            constructing the LinkML slot schema contained in the current instance

        Returns:
            A dictionary containing the mapping of `CoreSchemaOrFieldType` to a
                handler method for constructing the LinkML slot schema for that type.

        Raises:
            TypeError: If no method has been defined for constructing the slot schema
                for one of the schema or field types
        """
        mapping: dict[CoreSchemaOrFieldType, Callable[[CoreSchemaOrField], None]] = {}
        core_schema_types: list[
            CoreSchemaOrFieldType
        ] = _typing_extra.all_literal_values(
            CoreSchemaOrFieldType  # type: ignore
        )
        for key in core_schema_types:
            method_name = f"_{key.replace('-', '_')}_schema"
            try:
                mapping[key] = getattr(self, method_name)
            except AttributeError as e:  # pragma: no cover
                raise TypeError(
                    f"No method for constructing the slot schema for "
                    f"core_schema.type={key!r} "
                    f"(expected: {type(self).__name__}.{method_name})"
                ) from e
        return mapping

    def generate(self) -> SlotDefinition:
        """
        Generate a LinkML slot schema from the Pydantic model field schema provided to
            this generator.

        :return: The generated LinkML slot schema
        """
        if self._used:
            raise UserError(
                f"This {type(self).__name__} instance has already been used to generate"
                " a slot schema. You must create a new instance of "
                f"{type(self).__name__} to generate another schema."
            )

        # Initialized the `required` meta slot to `True` since all
        # Pydantic fields are required unless a default value is provided
        self._slot.required = True

        # Shape the contained slot according to core schema of the corresponding field
        self._shape_slot(self._field_schema.schema)

        self._used = True
        return self._slot

    def _shape_slot(self, schema: CoreSchemaOrField) -> None:
        """
        Shape the slot definition contained in this generator
            per the schema provided

        Note:
             This method is inspired by
                `pydantic.json_schema.GenerateJsonSchema.generate_inner()`
        """
        shape_slot_for_specific_schema_type = self._schema_type_to_method[
            schema["type"]
        ]
        shape_slot_for_specific_schema_type(schema)

    def _attach_note(self, note: str) -> None:
        """
        Attach a note to the contained slot definition

        :param note: The note to attach
        """
        self._slot.notes.append(f"{__package__}: {note}")

    def _any_schema(self, _schema: core_schema.AnySchema) -> None:
        """
        Shape the contained slot definition to match any value

        :param _schema: The core schema
        """
        self._slot.range = any_class_def.name

    def _none_schema(self, _schema: core_schema.NoneSchema) -> None:
        """
        Shape the contained slot definition to match `core_schema.NoneSchema`

        :param _schema: The `core_schema.NoneSchema` representing the `None` value
            restriction

        Note in the contained slot definition that the corresponding field in
        a Pydantic model is restricted to `NoneType` yet LinkML does not have
        null values

        Note: Currently, this method does not add any restriction to the contained slot.
        """
        self._attach_note(
            "LinkML does not have null values. "
            "(For details, see https://github.com/orgs/linkml/discussions/1975)."
        )

    def _bool_schema(self, _schema: core_schema.BoolSchema) -> None:
        """
        Shape the contained slot definition to match a Boolean value

        :param _schema: The `core_schema.BoolSchema` representing the boolean value
            restriction
        """
        self._slot.range = "boolean"

    def _int_schema(self, schema: core_schema.IntSchema) -> None:
        """
        Shape the contained slot definition to match an integer value

        :param schema: The `core_schema.IntSchema` representing the integer value
            restriction
        """
        self._slot.range = "integer"

        if "multiple_of" in schema:
            self._attach_note(
                "Unable to express the restriction of being "
                f"a multiple of {schema['multiple_of']}."
            )
        if "le" in schema:
            self._slot.maximum_value = schema["le"]
        if "ge" in schema:
            self._slot.minimum_value = schema["ge"]
        if "lt" in schema:
            self._slot.maximum_value = (
                schema["lt"] - 1
                if self._slot.maximum_value is None
                else min(self._slot.maximum_value, schema["lt"] - 1)
            )
        if "gt" in schema:
            self._slot.minimum_value = (
                schema["gt"] + 1
                if self._slot.minimum_value is None
                else max(self._slot.minimum_value, schema["gt"] + 1)
            )

    # noinspection DuplicatedCode
    def _float_schema(self, schema: core_schema.FloatSchema) -> None:
        """
        Shape the contained slot definition to match a float value

        :param schema: The `core_schema.FloatSchema` representing the float value
            restriction
        """
        self._slot.range = "float"
        if "allow_inf_nan" not in schema or schema["allow_inf_nan"]:
            self._attach_note(
                "LinkML does not have support for `'+inf'`, `'-inf'`, and `'NaN'` "
                "values. Support for these values is not translated."
            )
        if "multiple_of" in schema:
            self._attach_note(
                "Unable to express the restriction of being "
                f"a multiple of {schema['multiple_of']}."
            )
        if "le" in schema:
            self._slot.maximum_value = schema["le"]
        if "ge" in schema:
            self._slot.minimum_value = schema["ge"]
        if "lt" in schema:
            self._attach_note(
                f"Unable to express the restriction of being less than {schema['lt']}. "
                f"For details, see https://github.com/orgs/linkml/discussions/2144"
            )
        if "gt" in schema:
            self._attach_note(
                f"Unable to express the restriction of being greater than "
                f"{schema['gt']}. "
                f"For details, see https://github.com/orgs/linkml/discussions/2144"
            )

    # noinspection DuplicatedCode
    def _decimal_schema(self, schema: core_schema.DecimalSchema) -> None:
        """
        Shape the contained slot definition to match a decimal value

        :param schema: The `core_schema.DecimalSchema` representing the decimal value
            restriction
        """
        self._slot.range = "decimal"

        if "allow_inf_nan" in schema and schema["allow_inf_nan"]:
            self._attach_note(
                "LinkML does not have support for `'+inf'`, `'-inf'`, and `'NaN'` "
                "values. Support for these values is not translated."
            )
        if "multiple_of" in schema:
            self._attach_note(
                "Unable to express the restriction of being "
                f"a multiple of {schema['multiple_of']}."
            )
        if "le" in schema:
            self._slot.maximum_value = schema["le"]
        if "ge" in schema:
            self._slot.minimum_value = schema["ge"]
        if "lt" in schema:
            self._attach_note(
                f"Unable to express the restriction of being less than {schema['lt']}. "
                f"For details, see https://github.com/orgs/linkml/discussions/2144"
            )
        if "gt" in schema:
            self._attach_note(
                f"Unable to express the restriction of being greater than "
                f"{schema['gt']}. "
                f"For details, see https://github.com/orgs/linkml/discussions/2144"
            )
        if "max_digits" in schema:
            self._attach_note(
                "Unable to express the restriction of max number "
                f"of {schema['max_digits']} digits within a `Decimal` value."
            )
        if "decimal_places" in schema:
            self._attach_note(
                "Unable to express the restriction of max number of "
                f"{schema['decimal_places']} decimal places within a `Decimal` value."
            )

    def _str_schema(self, schema: core_schema.StringSchema) -> None:
        """
        Shape the contained slot definition to match a string value

        :param schema: The `core_schema.StringSchema` representing the string value
            restriction
        """
        self._slot.range = "string"

        if "pattern" in schema:
            self._slot.pattern = schema["pattern"]

        max_length: Optional[int] = schema.get("max_length")
        min_length: Optional[int] = schema.get("min_length")

        if max_length is not None:
            self._attach_note(
                "LinkML does not have direct support for max length constraints. "
                f"The max length constraint of {max_length} is incorporated "
                "into the pattern of the slot."
            )

        if min_length is not None:
            self._attach_note(
                "LinkML does not have direct support for min length constraints. "
                f"The min length constraint of {min_length} is incorporated "
                "into the pattern of the slot."
            )

        # == Incorporate any length constraints into the pattern of the slot ==
        if max_length is not None or min_length is not None:
            length_constraint_regex = (
                f"^(?=."
                f"{{{min_length if min_length is not None else ''},"
                f"{max_length if max_length is not None else ''}}}$)"
            )

            orig_ptrn = self._slot.pattern
            if orig_ptrn is not None:
                # == There is an existing pattern carried over
                # from the Pydantic core schema ==

                # Update the pattern to include the length constraint
                self._slot.pattern = (
                    f"{length_constraint_regex}"
                    f"{orig_ptrn[1:] if orig_ptrn.startswith('^') else orig_ptrn}"
                )
            else:
                # == There is no existing pattern carried over
                # from the Pydantic core schema ==

                # Set the pattern to the length constraint
                self._slot.pattern = length_constraint_regex

        if "strip_whitespace" in schema and schema["strip_whitespace"]:
            self._attach_note(
                "Unable to express the option of "
                "stripping leading and trailing whitespace in LinkML."
            )
        if "to_lower" in schema and schema["to_lower"]:
            self._attach_note(
                "Unable to express the option of converting the string to lowercase "
                "in LinkML."
            )
        if "to_upper" in schema and schema["to_upper"]:
            self._attach_note(
                "Unable to express the option of converting the string to uppercase "
                "in LinkML."
            )
        if "regex_engine" in schema:
            # I believe nothing needs to be done here.
            # The regex engine mostly supports a subset of the standard regular
            # expressions. For more info,
            # see https://docs.pydantic.dev/latest/migration/#patterns-regex-on-strings.
            pass

    def _bytes_schema(self, schema: core_schema.BytesSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _date_schema(self, schema: core_schema.DateSchema) -> None:
        """
        Shape the contained slot definition to match a date value

        :param schema: The `core_schema.DateSchema` representing the date value
            restriction
        """
        self._slot.range = "date"

        if "le" in schema:
            self._attach_note(
                "Unable to express the restriction of being less than or equal to "
                "a date. LinkML lacks direct support for this restriction."
            )
        if "ge" in schema:
            self._attach_note(
                "Unable to express the restriction of being greater than or equal to "
                "a date. LinkML lacks direct support for this restriction."
            )
        if "lt" in schema:
            self._attach_note(
                "Unable to express the restriction of being less than a date. "
                "LinkML lacks direct support for this restriction."
            )
        if "gt" in schema:
            self._attach_note(
                "Unable to express the restriction of being greater than a date. "
                "LinkML lacks direct support for this restriction."
            )
        if "now_op" in schema:
            self._attach_note(
                "Unable to express the restriction of being before or after the "
                "current date. LinkML lacks direct support for this restriction."
            )
        if "now_utc_offset" in schema:
            self._attach_note(
                "Unable to express the utc offset of the current date "
                "in the restriction of being before or after the current date. "
                "LinkML lacks direct support for this restriction."
            )

    def _time_schema(self, schema: core_schema.TimeSchema) -> None:
        """
        Shape the contained slot definition to match a time value

        :param schema: The `core_schema.TimeSchema` representing the time value
            restriction
        """
        self._slot.range = "time"

        if "le" in schema:
            self._attach_note(
                "Unable to express the restriction of being less than or equal to "
                "a time. LinkML lacks direct support for this restriction."
            )
        if "ge" in schema:
            self._attach_note(
                "Unable to express the restriction of being greater than or equal to "
                "a time. LinkML lacks direct support for this restriction."
            )
        if "lt" in schema:
            self._attach_note(
                "Unable to express the restriction of being less than a time. "
                "LinkML lacks direct support for this restriction."
            )
        if "gt" in schema:
            self._attach_note(
                "Unable to express the restriction of being greater than a time. "
                "LinkML lacks direct support for this restriction."
            )
        if "tz_constraint" in schema:
            self._attach_note(
                f"Unable to express the timezone constraint of "
                f"{schema['tz_constraint']}. "
                f"LinkML lacks direct support for this restriction."
            )
        if "microseconds_precision" in schema:
            self._attach_note(
                f"Unable to express the microseconds precision constraint of "
                f"{schema['microseconds_precision']}. "
                "LinkML lacks direct support for this restriction."
            )

    def _datetime_schema(self, schema: core_schema.DatetimeSchema) -> None:
        """
        Shape the contained slot definition to match a datetime value

        :param schema: The `core_schema.DatetimeSchema` representing the datetime value
            restriction
        """
        self._slot.range = "datetime"

        if "le" in schema:
            self._attach_note(
                "Unable to express the restriction of being less than or equal to "
                "a datetime. LinkML lacks direct support for this restriction."
            )
        if "ge" in schema:
            self._attach_note(
                "Unable to express the restriction of being greater than or equal to "
                "a datetime. LinkML lacks direct support for this restriction."
            )
        if "lt" in schema:
            self._attach_note(
                "Unable to express the restriction of being less than a datetime. "
                "LinkML lacks direct support for this restriction."
            )
        if "gt" in schema:
            self._attach_note(
                "Unable to express the restriction of being greater than a datetime. "
                "LinkML lacks direct support for this restriction."
            )
        if "now_op" in schema:
            self._attach_note(
                "Unable to express the restriction of being before or after the "
                "current datetime. LinkML lacks direct support for this restriction."
            )
        if "tz_constraint" in schema:
            self._attach_note(
                f"Unable to express the timezone constraint of "
                f"{schema['tz_constraint']}. "
                f"LinkML lacks direct support for this restriction."
            )
        if "now_utc_offset" in schema:
            self._attach_note(
                "Unable to express the utc offset of the current datetime in "
                "the restriction of being before or after the current datetime. "
                "LinkML lacks direct support for this restriction."
            )
        if "microseconds_precision" in schema:
            self._attach_note(
                f"Unable to express the microseconds precision constraint of "
                f"{schema['microseconds_precision']}. "
                "LinkML lacks direct support for this restriction."
            )

    def _timedelta_schema(self, schema: core_schema.TimedeltaSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _literal_schema(self, schema: core_schema.LiteralSchema) -> None:
        """
        Shape the contained slot definition to allow only a specific set of values

        :param schema: The `core_schema.LiteralSchema` representing a set of literal
            values that the slot can take
        """
        # Check if the types of the given literals are supportable
        expected: list[Any] = schema["expected"]
        literal_types = {type(literal) for literal in expected}
        if not literal_types.issubset({str, int}):
            self._attach_note(
                "Unable to express the restriction of being one of the elements in "
                f"`{expected}`. LinkML has direct support for only string "
                f"and integer elements in expressing such a restriction."
            )
        else:
            self._slot.range = "Any"
            self._slot.any_of = [
                (
                    AnonymousSlotExpression(equals_string=literal, range="string")
                    if type(literal) is str
                    else AnonymousSlotExpression(equals_number=literal, range="integer")
                )
                for literal in expected
            ]

    def _enum_schema(self, schema: core_schema.EnumSchema) -> None:
        """
        Shape the contained slot definition to match an enum value

        :param schema: The `core_schema.EnumSchema` representing the enum type the
            value belongs to
        """
        enum_name = schema["cls"].__name__

        self._slot.range = enum_name
        if "missing" in schema:
            self._attach_note(
                f"Unable to express calling {schema['missing'].__name__} in LinkML "
                f"when the provide value is not found in the enum type, {enum_name}."
            )

    def _is_instance_schema(self, schema: core_schema.IsInstanceSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _is_subclass_schema(self, schema: core_schema.IsSubclassSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _callable_schema(self, schema: core_schema.CallableSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _list_schema(self, schema: core_schema.ListSchema) -> None:
        """
        Shape the contained slot definition to match a list value

        :param schema: The `core_schema.ListSchema` representing
            the list value restriction
        """
        if self._slot.multivalued:
            # === This must be a nested list type ===
            self._attach_note(
                "Translation is incomplete." "Nested list types are not yet supported."
            )
            return

        self._slot.multivalued = True
        if "min_length" in schema:
            self._slot.minimum_cardinality = schema["min_length"]
        if "max_length" in schema:
            self._slot.maximum_cardinality = schema["max_length"]
        if "items_schema" in schema:
            self._shape_slot(schema["items_schema"])

    def _tuple_schema(self, schema: core_schema.TupleSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _set_schema(self, schema: core_schema.SetSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _frozenset_schema(self, schema: core_schema.FrozenSetSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _generator_schema(self, schema: core_schema.GeneratorSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _dict_schema(self, schema: core_schema.DictSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _function_schema(
        self,
        schema: Union[
            core_schema.AfterValidatorFunctionSchema,
            core_schema.BeforeValidatorFunctionSchema,
            core_schema.WrapValidatorFunctionSchema,
            core_schema.PlainValidatorFunctionSchema,
        ],
    ) -> None:
        """
        A helper method that shapes the contained slot definition to provide
            the restriction set by a validation function

        :param schema: The schema representing the validation function
        """
        mode = schema["type"].split("-")[1]

        self._attach_note(
            "Unable to translate the logic contained in "
            f"the {mode} validation function, {schema['function']['function']!r}."
        )
        if mode != "plain":
            self._shape_slot(schema["schema"])

    def _function_after_schema(
        self, schema: core_schema.AfterValidatorFunctionSchema
    ) -> None:
        """
        Shape the contained slot definition to provide the restriction set by an after
            validation function

        :param schema: The schema representing the after validation function
        """
        self._function_schema(schema)

    def _function_before_schema(
        self, schema: core_schema.BeforeValidatorFunctionSchema
    ) -> None:
        """
        Shape the contained slot definition to provide the restriction set by a before
            validation function

        :param schema: The schema representing the before validation function
        """
        self._function_schema(schema)

    def _function_wrap_schema(
        self, schema: core_schema.WrapValidatorFunctionSchema
    ) -> None:
        """
        Shape the contained slot definition to provide the restriction set by a wrap
            validation function

        :param schema: The schema representing the wrap validation function
        """
        self._function_schema(schema)

    def _function_plain_schema(
        self, schema: core_schema.PlainValidatorFunctionSchema
    ) -> None:
        """
        Shape the contained slot definition to provide the restriction set by a plain
            validation function

        :param schema: The schema representing the plain validation function
        """
        self._function_schema(schema)

    def _default_schema(self, schema: core_schema.WithDefaultSchema) -> None:
        """
        Shape the contained slot definition to have a default value

        :param schema: The `core_schema.WithDefaultSchema` representing the default
            value specification
        """
        inner_schema = schema["schema"]

        self._slot.required = False
        if "default" in schema and (default := schema["default"]) is not None:
            # === Set `ifabsent` meta slot ===
            default_type = type(default)
            if default_type is bool:
                self._slot.ifabsent = str(default)
            elif default_type is int:
                self._slot.ifabsent = f"int({default})"
            elif default_type is str:
                self._slot.ifabsent = f"string({default})"
            elif default_type is float:
                self._slot.ifabsent = f"float({default})"
            elif default_type is date:
                self._slot.ifabsent = f"date({default})"
            else:
                self._attach_note(
                    f"Unable to set a default value of {default!r} in LinkML. "
                    f"Default values of type {default_type} are not supported."
                )

            if inner_schema["type"] == "nullable":
                self._attach_note(
                    "Warning: LinkML doesn't have a null value. "
                    "The translation of `Optional` in Python may need further "
                    "adjustments."
                )
        if "default_factory" in schema:
            self._attach_note(
                "Unable to express the default factory, "
                f"{schema['default_factory']!r}, in LinkML."
            )
        if "on_error" in schema and schema["on_error"] != "raise":
            self._attach_note(
                "Unable to express the `on_error` option of "
                f"{schema['on_error']} in LinkML."
            )
        if "validate_default" in schema:
            # This is purposely left empty.
            # LinkML validates the default value of a slot, provided by the `ifabsent`
            # meta slot, no matter what. In the case of `schema['validate_default']`
            # being `False`, the default, LinkML's behavior is just stricter, and
            # attaching a note to the slot about not able to express
            # `schema['validate_default']` being `False` would generate too much
            # clutter.
            pass

        self._shape_slot(inner_schema)

    def _nullable_schema(self, schema: core_schema.NullableSchema) -> None:
        """
        Shape the contained slot definition to match a nullable value restriction

        :param schema: The schema representing the nullable value restriction

        Note: There is no null value in LinkML
              (https://github.com/orgs/linkml/discussions/1975).
        """
        if self._slot.required:
            # === The field being translated must have no default value ===

            self._attach_note(
                "Warning: LinkML doesn't have a null value. "
                "The translation of `Optional` for a required field may require "
                "further adjustments."
            )

        # Note: The case of `self._slot.required` being `False` is handled in
        #   `SlotGenerator._default_schema()

        self._shape_slot(schema["schema"])

    def _union_schema(self, schema: core_schema.UnionSchema) -> None:
        """
        Shape the contained slot definition to match a union restriction

        :param schema: The schema representing the union restriction
        """
        # todo: the current implementation is just an annotation
        #   A usable implementation is yet to be decided. Useful information
        #   can be found at, https://github.com/orgs/linkml/discussions/2154
        self._attach_note(
            "Warning: The translation is incomplete. Union types are yet to be "
            "supported."
        )

    def _tagged_union_schema(self, schema: core_schema.TaggedUnionSchema) -> None:
        """
        Shape the contained slot definition to match a tagged union restriction

        :param schema: The schema representing the tagged union restriction
        """
        # todo: the current implementation is just an annotation
        #   A usable implementation is yet to be decided. Useful information
        #   can be found at, https://github.com/orgs/linkml/discussions/2154
        #   and https://linkml.io/linkml/schemas/type-designators.html
        self._attach_note(
            "Warning: The translation is incomplete. Tagged union types are yet to be "
            "supported."
        )

    def _chain_schema(self, schema: core_schema.ChainSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _lax_or_strict_schema(self, schema: core_schema.LaxOrStrictSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _json_or_python_schema(self, schema: core_schema.JsonOrPythonSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _typed_dict_schema(self, schema: core_schema.TypedDictSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _model_fields_schema(self, schema: core_schema.ModelFieldsSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _model_schema(self, schema: core_schema.ModelSchema) -> None:
        """
        Shape the contained slot definition to match an instance of a model, or class
            in LinkML

        :param schema: The schema representing the model
        """
        self._slot.range = schema["cls"].__name__

    def _dataclass_args_schema(self, schema: core_schema.DataclassArgsSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _dataclass_schema(self, schema: core_schema.DataclassSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _arguments_schema(self, schema: core_schema.ArgumentsSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _call_schema(self, schema: core_schema.CallSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _custom_error_schema(self, schema: core_schema.CustomErrorSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _json_schema(self, schema: core_schema.JsonSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _url_schema(self, schema: core_schema.UrlSchema) -> None:
        """
        Shape the contained slot definition to match a URL restriction

        :param schema: The schema representing the URL restriction
        """
        # This method may be further improved or implemented more fully upon the
        # resolution of https://github.com/linkml/linkml/issues/2215

        self._slot.range = "uri"

        # Incorporate `max_length` and `allowed_schemes` restrictions into the pattern
        # meta slot
        max_length: Optional[int] = schema.get("max_length")
        allowed_schemes: Optional[list[str]] = schema.get("allowed_schemes")
        max_length_re = rf"(?=.{{,{max_length}}}$)" if max_length is not None else ""
        allowed_schemes_re = (
            rf"(?i:{'|'.join(re.escape(scheme) for scheme in allowed_schemes)})"
            if allowed_schemes is not None
            else r"[^\s]+"
        )
        self._slot.pattern = rf"^{max_length_re}{allowed_schemes_re}://[^\s]+$"

        if "host_required" in schema:
            self._attach_note("Unable to express the `host_required` option in LinkML.")
        if "default_host" in schema:
            self._attach_note("Unable to express the `default_host` option in LinkML.")
        if "default_port" in schema:
            self._attach_note("Unable to express the `default_port` option in LinkML.")
        if "default_path" in schema:
            self._attach_note("Unable to express the `default_path` option in LinkML.")

    def _multi_host_url_schema(self, schema: core_schema.MultiHostUrlSchema) -> None:
        raise NotImplementedError("Method not yet implemented")

    def _definitions_schema(self, schema: core_schema.DefinitionsSchema) -> None:
        """
        Shape the contained slot definition to match a `core_schema.DefinitionsSchema`

        :param schema: The `core_schema.DefinitionsSchema`
        """
        self._shape_slot(resolve_ref_schema(schema, self._field_schema.context))

    def _definition_ref_schema(
        self, schema: core_schema.DefinitionReferenceSchema
    ) -> None:
        """
        Shape the contained slot definition to match
        a `core_schema.DefinitionReferenceSchema`

        :param schema: The `core_schema.DefinitionsSchema`
        """
        self._shape_slot(resolve_ref_schema(schema, self._field_schema.context))

    def _uuid_schema(self, schema: core_schema.UuidSchema) -> None:
        """
        Shape the contained slot definition to match a UUID restriction

        :param schema: The schema representing the UUID restriction
        """
        self._slot.range = "string"
        self._slot.pattern = get_uuid_regex(schema.get("version"))

    def _model_field_schema(self, schema: core_schema.ModelField) -> None:

        raise NotImplementedError("Method not yet implemented")

    def _dataclass_field_schema(self, schema: core_schema.DataclassField) -> None:

        raise NotImplementedError("Method not yet implemented")

    def _typed_dict_field_schema(self, schema: core_schema.TypedDictField) -> None:

        raise NotImplementedError("Method not yet implemented")

    def _computed_field_schema(self, schema: core_schema.ComputedField) -> None:
        raise NotImplementedError("Method not yet implemented")
