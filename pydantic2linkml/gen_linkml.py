from enum import Enum
from typing import Optional
from collections.abc import Iterable

from pydantic import BaseModel
from linkml_runtime.utils.schema_builder import SchemaBuilder
from linkml_runtime.linkml_model import (
    SchemaDefinition,
    EnumDefinition,
    PermissibleValue,
)

from .exceptions import UserError
from .tools import ensure_unique_names


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

        self._models = models
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

        # Add enums to the schema
        self._add_enums()

        # Add slots to the schema
        # Add classes to the schema
        # Make sure to provide slot usage in the individual classes if needed

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
                    permissible_values=[
                        PermissibleValue(text=value_str, meaning=value_str)
                        for value_str in enum_value_strs
                    ],
                )
            )
