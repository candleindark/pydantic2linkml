from __future__ import annotations

from typing import Type, NamedTuple, Optional, cast
import re

from pydantic import BaseModel
from pydantic_core import core_schema

from .exceptions import NameCollisionError


class LocallyDefinedFields(NamedTuple):
    new: dict[str, FieldSchema]
    overriding: dict[str, FieldSchema]


class FieldSchema(NamedTuple):
    # The resolved Pydantic core schema of the field
    schema: core_schema.CoreSchema

    # The context in which the Pydantic core schema of the field was defined
    # (i.e. the Pydantic core schema of the model that defined the field).
    # This context is needed to resolve any references in the field schema.
    context: core_schema.CoreSchema


def get_parent_model(model: Type[BaseModel]) -> Optional[Type[BaseModel]]:
    """
    Get the parent Pydantic model of a Pydantic model

    :param model: The Pydantic model
    :return: The parent Pydantic model of the input model. Returns `None` if the input
        is `pydantic.BaseModel`.
    :raises ValueError: If the model has multiple Pydantic models as an immediate parent

    Note: This function only handles Pydantic models having only one Pydantic model as
        an immediate parent. If the model has multiple Pydantic models as an immediate
        parent, this function will raise a ValueError.
    """
    parent_model: Optional[Type[BaseModel]] = None

    for base in model.__bases__:
        if issubclass(base, BaseModel):
            if parent_model is None:
                parent_model = base
            else:
                raise ValueError(f"Model {model} has multiple Pydantic base models")

    return parent_model


def resolve_ref_schema(
    maybe_ref_schema: core_schema.CoreSchema,
    context: core_schema.CoreSchema,
) -> core_schema.CoreSchema:
    """
    Resolves reference in the core schema.

    :param maybe_ref_schema: A `CoreSchema` object that's possibly a reference,
        i.e. a `DefinitionsSchema` or a `DefinitionReferenceSchema`.
    :param context: A `CoreSchema` in which the `maybe_ref_schema` is defined.
        This can be the same object as `maybe_ref_schema`.
    :return: The resolved `CoreSchema` object.

    :raises ValueError: If `context` is not a `DefinitionsSchema` object when
        `maybe_ref_schema` is a `DefinitionsSchema` or `DefinitionReferenceSchema`.
    :raises RuntimeError: If the referenced schema is not found in the provided context.

    Note:
        This function mimics `resolve_ref_schema` in
        `pydantic._internal._schema_generation_shared.CallbackGetCoreSchemaHandler`
    """
    schema_type = maybe_ref_schema["type"]

    if schema_type == "definitions" or schema_type == "definition-ref":
        if context["type"] != "definitions":
            raise ValueError(
                "`context` must be a `DefinitionsSchema` object when "
                "`maybe_ref_schema` is a `DefinitionsSchema` "
                "or `DefinitionReferenceSchema`."
            )

    context = cast(core_schema.DefinitionsSchema, context)

    if schema_type == "definition-ref":
        ref = maybe_ref_schema["schema_ref"]
        for schema in context["definitions"]:
            if schema["ref"] == ref:
                return schema
        else:
            raise RuntimeError(
                f"Referenced schema by {ref} not found in provided context"
            )
    elif schema_type == "definitions":
        return resolve_ref_schema(maybe_ref_schema["schema"], context)
    return maybe_ref_schema


def get_field_schema(model: Type[BaseModel], fn: str) -> core_schema.CoreSchema:
    """
    Get the resolved Pydantic core schema of a field in a Pydantic model

    :param model: The Pydantic model
    :param fn: The name of the field
    :return: The Pydantic core schema of the field

    Note: The returned schema is guaranteed to be resolved, i.e. it is not a reference
        schema.
    """
    context = model.__pydantic_core_schema__
    model_schema = resolve_ref_schema(context, context)

    assert model_schema["type"] == "model"

    if model_schema["schema"]["type"] == "model-fields":
        return resolve_ref_schema(
            cast(core_schema.ModelFieldsSchema, model_schema["schema"])["fields"][fn][
                "schema"
            ],
            context,
        )
    else:
        raise NotImplementedError(
            f"This function currently doesn't support the inner schema of "
            f"a `ModelSchema` being the type of \"{model_schema['schema']['type']}\""
        )


def get_locally_defined_fields(model: Type[BaseModel]) -> LocallyDefinedFields:
    """
    Get the fields defined in a Pydantic model that are not inherited

    :param model: The Pydantic model
    :return:
        A tuple of two dictionaries:
            The first contains the fields that are newly defined in this model as keys.
            The second contains the fields that are redefined (overriding) in this model
                as keys.
            The values in both dictionaries are `FieldSchema` objects representing the
                Pydantic core schemas of respective fields in context.
    """

    parent_model = get_parent_model(model)

    # Names of locally defined fields
    locally_defined_fn = set(model.model_fields).intersection(model.__annotations__)

    # Names newly defined fields
    new_fn = locally_defined_fn.difference(parent_model.model_fields)

    # Names of overriding fields
    overriding_fn = locally_defined_fn - new_fn

    model_schema = model.__pydantic_core_schema__
    return LocallyDefinedFields(
        new={
            fn: FieldSchema(get_field_schema(model, fn), model_schema) for fn in new_fn
        },
        overriding={
            fn: FieldSchema(get_field_schema(model, fn), model_schema)
            for fn in overriding_fn
        },
    )


def ensure_unique_names(*clses: Type) -> None:
    """
    In the context of the collection of all classes given as an argument,
    ensure all of them have a unique name.

    :param clses: The classes given as an argument packed in a tuple

    :raises NameCollisionError: If there are classes with the same name
    """
    # Sort classes into buckets by name
    buckets: dict[str, list[Type]] = {}
    for cls in clses:
        name = cls.__name__
        if name in buckets:
            buckets[name].append(cls)
        else:
            buckets[name] = [cls]

    # Build error message for any name collisions
    err_msg: Optional[str] = None
    for name, lst in buckets.items():
        if len(lst) > 1:
            new_err_msg = f"Name collision @ {name}: {lst!r}"
            err_msg = new_err_msg if err_msg is None else f"{err_msg}; {new_err_msg}"

    if err_msg is not None:
        raise NameCollisionError(err_msg)


def normalize_whitespace(text: str) -> str:
    """
    Return a version of the input text with leading and trailing whitespaces removed
    and sequences of consecutive whitespaces replaced with a single space.
    """
    return re.sub(r"\s+", " ", text.strip())
