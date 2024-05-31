from enum import Enum
from typing import Optional, Callable
from collections.abc import Iterable
from collections import defaultdict
from itertools import chain
from warnings import warn
from operator import itemgetter

from pydantic import BaseModel

# noinspection PyProtectedMember
from pydantic._internal import _typing_extra
from pydantic.json_schema import (
    CoreSchemaOrFieldType,
    CoreSchemaOrField,
)
from linkml_runtime.utils.schema_builder import SchemaBuilder
from linkml_runtime.linkml_model import (
    SchemaDefinition,
    EnumDefinition,
    PermissibleValue,
    SlotDefinition,
)

from .exceptions import UserError
from .tools import (
    ensure_unique_names,
    normalize_whitespace,
    get_locally_defined_fields,
    LocallyDefinedFields,
    FieldSchema,
    resolve_ref_schema,
    bucketize,
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
        add_defaults: bool = True,
    ):
        """
        :param name: The name of the LinkML schema to be generated
        :param id_: The ID of the LinkML schema to be generated
        :param models: An iterable of Pydantic models to be converted to LinkML classes
            in the generated schema
        :param enums: An iterable of Enums to be converted to LinkML enums in
            the generated schema
        :param add_defaults: Whether to set some defaults in the generated schema as
            specified in `SchemaBuilder.add_defaults`

        raises NameCollisionError: If there are classes with the same name in the
            combined collection of `models` and `enums`
        """
        ensure_unique_names(*models, *enums)

        # Map of models to their locally defined fields
        self._m_f_map: dict[type[BaseModel], LocallyDefinedFields] = {
            m: get_locally_defined_fields(m) for m in models
        }

        self._enums = enums

        sb = SchemaBuilder(name, id_)
        self._sb = sb.add_defaults() if add_defaults else sb

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
        for f_name, schema_lst in buckets.items():
            # Use the first schema in `schema_lst` to generate the slot
            slot = SlotGenerator(f_name, schema_lst[0]).generate()

            # Add the slot to the schema
            self._sb.add_slot(slot)

    def _add_classes(self):
        """
        Add the classes construed from the models in `self._m_f_map` to the schema
        """
        raise NotImplementedError("Method not yet implemented")
        # todo: Make sure to provide slot usage in the individual classes if needed


class SlotGenerator:
    """
    Instances of this class are single-use slot generators.

    Note:
        Each instance of this class should only be used once to generate
            a LinkML slot schema.
    """

    def __init__(self, field_name: str, field_schema: FieldSchema):
        """
        :param field_name: The name of the Pydantic model field corresponding
            to the slot to be generated
        :param field_schema: The `FieldSchema` object specifying the Pydantic core
            schema of the corresponding field with context
        """

        self._slot: SlotDefinition = SlotDefinition(name=field_name)
        self._field_schema: FieldSchema = field_schema
        self._schema_type_to_method = self.build_schema_type_to_method()

        # This changes to True after this generator generates a slot schema
        # (for preventing issues caused by accidental re-use
        # of this generator). See class docstring for more info.
        self._used: bool = False

    def build_schema_type_to_method(
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
        raise NotImplementedError("Method not yet implemented")
