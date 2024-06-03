from __future__ import annotations

import importlib
from typing import Type, NamedTuple, Optional, cast, TypeVar
from types import ModuleType
import re
from collections.abc import Iterable, Callable
from collections import defaultdict
from operator import attrgetter
from enum import Enum
import inspect

from pydantic import BaseModel, RootModel
from pydantic_core import core_schema

# noinspection PyProtectedMember
from pydantic._internal import _core_utils

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


def get_parent_models(model: Type[BaseModel]) -> list[Type[BaseModel]]:
    """
    Get the parent Pydantic models of a Pydantic model

    :param model: The Pydantic model
    :return: The list of parent Pydantic models of the input model

    Note: The order of the parent models returned is the models' order in the definition
        of the input model.
    Note: The input Pydantic model of `pydantic.BaseModel` produces a result of an empty
        list.
    """
    return [b for b in model.__bases__ if issubclass(b, BaseModel)]


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


def strip_function_schema(
    schema: _core_utils.AnyFunctionSchema,
) -> core_schema.CoreSchema:
    """
    Strip the outermost schema of a function schema

    :param schema: The function schema
    :return: The inner schema of the function schema
    :raises ValueError: If the given function schema is not a function with an inner
        schema
    """

    if _core_utils.is_function_with_inner_schema(schema):
        return schema["schema"]
    else:
        raise ValueError(
            "The given function schema is not a function with an inner schema. "
            "No outer schema to strip."
        )


# A mapping from unneeded wrapping schema types to functions that strip the outermost
# unneeded wrapping schema. The set of schema types deemed as unneeded may change in
# the future if we are able to harvest the information in any of the schema types.
UNNEEDED_WRAPPING_SCHEMA_TYPE_TO_STRIP_FUNC: dict[
    str, Callable[[core_schema.CoreSchema], core_schema.CoreSchema]
] = {
    "function-before": strip_function_schema,
    "function-after": strip_function_schema,
    "function-wrap": strip_function_schema,
    "function-plain": strip_function_schema,
}


def strip_unneeded_wrapping_schema(
    schema: core_schema.CoreSchema,
) -> core_schema.CoreSchema:
    """
    Strip the outermost unneeded wrapping schema

    :param schema: The schema to be stripped
    :return: The inner schema of the given schema if the outermost schema of the given
        schema is an unneeded wrapping schema. Otherwise, the given schema itself is
        returned.
    """
    schema_type = schema["type"]

    if schema_type in UNNEEDED_WRAPPING_SCHEMA_TYPE_TO_STRIP_FUNC:
        return UNNEEDED_WRAPPING_SCHEMA_TYPE_TO_STRIP_FUNC[schema_type](schema)
    else:
        return schema


def get_model_schema(model: Type[BaseModel]) -> core_schema.ModelSchema:
    """
    Get the corresponding `core_schema.ModelSchema` of a Pydantic model

    :param model: The Pydantic model
    """
    raw_model_schema = model.__pydantic_core_schema__
    model_schema = raw_model_schema

    while True:
        model_schema = resolve_ref_schema(model_schema, context=raw_model_schema)

        # Strip an unneeded wrapping schema
        inner_schema = strip_unneeded_wrapping_schema(model_schema)

        if inner_schema is model_schema:
            # Exit while-loop if no stripping is done, i.e. `model_schema` is already
            # devoid of any unneeded wrapping schema
            break
        else:
            model_schema = inner_schema

    assert (
        model_schema["type"] == "model"
    ), "Assumption about how model schema is stored is wrong."

    return cast(core_schema.ModelSchema, model_schema)


def get_field_schema(model: Type[BaseModel], fn: str) -> core_schema.CoreSchema:
    """
    Get the resolved Pydantic core schema of a field in a Pydantic model

    :param model: The Pydantic model
    :param fn: The name of the field
    :return: The Pydantic core schema of the field

    Note: The returned schema is guaranteed to be resolved, i.e. it is not a reference
        schema.
    """

    model_schema = get_model_schema(model)

    if model_schema["schema"]["type"] == "model-fields":
        return resolve_ref_schema(
            cast(core_schema.ModelFieldsSchema, model_schema["schema"])["fields"][fn][
                "schema"
            ],
            context=model.__pydantic_core_schema__,
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
    # Names of locally defined fields
    locally_defined_fns = set(model.model_fields).intersection(model.__annotations__)

    # Names of newly defined fields
    new_fns = locally_defined_fns.difference(
        *(pm.model_fields for pm in get_parent_models(model))
    )

    # Names of overriding fields
    overriding_fns = locally_defined_fns - new_fns

    model_schema = model.__pydantic_core_schema__
    return LocallyDefinedFields(
        new={
            fn: FieldSchema(get_field_schema(model, fn), model_schema) for fn in new_fns
        },
        overriding={
            fn: FieldSchema(get_field_schema(model, fn), model_schema)
            for fn in overriding_fns
        },
    )


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def bucketize(
    items: Iterable[T],
    key_func: Callable[[T], K],
    value_func: Optional[Callable[[T], V]] = None,
) -> defaultdict[K, list[V]]:
    """
    Bucketize items based on a key function

    :param items: The items to bucketize
    :param key_func: The key function
    :param value_func: An optional function to transform the items before storing to
        the corresponding buckets identified by the corresponding keys
    :return: A dictionary with keys as the results of the key function and values as
        the list of (transformed) items that have the corresponding key
    """
    buckets: defaultdict[K, list[T]] = defaultdict(list)
    for item in items:
        key = key_func(item)
        buckets[key].append(item if value_func is None else value_func(item))
    return buckets


def ensure_unique_names(*clses: Type) -> None:
    """
    In the context of the collection of all classes given as an argument,
    ensure all of them have a unique name.

    :param clses: The classes given as an argument packed in a tuple

    :raises NameCollisionError: If there are classes with the same name
    """
    # Sort classes into buckets by name
    buckets: dict[str, list[type]] = bucketize(clses, attrgetter("__name__"))

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


# todo: write tests for this function
def get_all_modules(root_module_name: str) -> list[ModuleType]:
    """
    Get all modules that are recursively imported by when importing a given module

    :param root_module_name: The name of the given module
    :return: The list of imported modules
    """

    def _get_all_modules(_imported_modules: list[ModuleType], _root_module_name: str):
        try:
            module = importlib.import_module(_root_module_name)
            _imported_modules.append(module)
            for (
                submodule_filename
            ) in module.__loader__.get_resource_reader().contents():
                if submodule_filename.endswith(
                    ".py"
                ) and not submodule_filename.startswith("__"):
                    _get_all_modules(
                        _imported_modules,
                        f'{_root_module_name}.{submodule_filename[:-len(".py")]}',
                    )
        except ModuleNotFoundError:
            return _imported_modules
        return _imported_modules

    imported_modules: list[ModuleType] = []
    return _get_all_modules(imported_modules, root_module_name)


def fetch_defs(
    modules: Iterable[ModuleType],
) -> tuple[list[type[BaseModel]], list[type[Enum]]]:
    """
    Fetch Python objects that provide schema definitions from given modules

    :param modules: The given modules
    :return: A tuple of two lists:
        The first list contains strict subclasses of `pydantic.BaseModel` that is not
            a subclass of `pydantic.RootModel` in the given modules
        The second list contains strict subclasses of `enum.Enum` in the given modules
    """

    models: list[type[BaseModel]] = []
    enums: list[type[Enum]] = []

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):

            if (
                issubclass(cls, BaseModel)
                and cls is not BaseModel
                and not issubclass(cls, RootModel)
            ):
                models.append(cls)
            elif issubclass(cls, Enum) and cls is not Enum:
                enums.append(cls)

    return models, enums
