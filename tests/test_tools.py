from __future__ import annotations

import re
from enum import Enum, auto
from operator import itemgetter
from typing import ClassVar, Optional, Type, cast

import pytest
from pydantic import BaseModel, RootModel
from pydantic_core import core_schema

from pydantic2linkml.exceptions import NameCollisionError
from pydantic2linkml.tools import (
    bucketize,
    ensure_unique_names,
    fetch_defs,
    force_to_set,
    get_field_schema,
    get_locally_defined_fields,
    get_parent_models,
    get_uuid_regex,
    normalize_whitespace,
    resolve_ref_schema,
)


def test_get_parent_model():
    class Foo:
        pass

    class Bar:
        pass

    class Baz:
        pass

    class A(BaseModel):
        pass

    class B(Foo, A):
        pass

    class C(B, Bar):
        pass

    class X(BaseModel):
        pass

    class Z(X, Baz, C):
        pass

    assert len(get_parent_models(BaseModel)) == 0
    assert get_parent_models(A) == [BaseModel]
    assert get_parent_models(B) == [A]
    assert get_parent_models(C) == [B]
    assert get_parent_models(Z) == [X, C]


class TestResolveRefSchema:
    def test_valid_input(self):
        """
        Test with valid input
        """

        class A(BaseModel):
            pass

        class B(BaseModel):
            x: A
            y: Optional[A]
            z: B

        a_schema = A.__pydantic_core_schema__
        b_schema = B.__pydantic_core_schema__

        assert_err_msg = (
            "Wrong assumption about Pydantic behavior. Please re-write the test."
        )

        # If these two assertions fail, it doesn't mean `resolve_ref_schema` is wrong.
        # It only means the assumption on how Pydantic represents models in `CoreSchema`
        # is wrong, and we have to find another way to test `resolve_ref_schema`.
        assert a_schema["type"] == "model", assert_err_msg
        assert b_schema["type"] == "definitions", assert_err_msg

        resolved_a_schema = resolve_ref_schema(a_schema, a_schema)
        resolved_b_schema = resolve_ref_schema(b_schema, b_schema)
        assert resolved_a_schema["type"] == "model"
        assert resolved_a_schema["cls"] is A
        assert resolved_b_schema["type"] == "model"
        assert resolved_b_schema["cls"] is B

        x_field_schema = cast(
            core_schema.DefinitionReferenceSchema,
            cast(core_schema.ModelFieldsSchema, resolved_b_schema["schema"])["fields"][
                "x"
            ]["schema"],
        )

        # If this assertion fails, it doesn't mean `resolve_ref_schema` is wrong.
        # It only means the assumption about how Pydantic uses `definition-ref` is
        # wrong, and we have to find another way to test `resolve_ref_schema`.
        assert x_field_schema["type"] == "definition-ref", assert_err_msg

        assert (
            resolve_ref_schema(
                x_field_schema,
                b_schema,
            )["cls"]
            is A
        )

    def test_invalid_input(self):
        """
        Test with invalid input, i.e. context` is not a `DefinitionsSchema` object when
        `maybe_ref_schema` is a `DefinitionsSchema` or `DefinitionReferenceSchema`.
        """

        class A(BaseModel):
            pass

        class B(A):
            x: A
            y: B

        with pytest.raises(
            ValueError, match="`context` must be a `DefinitionsSchema` object"
        ):
            resolve_ref_schema(B.__pydantic_core_schema__, A.__pydantic_core_schema__)

    def test_missing_definition(self):
        """
        Test the case where the provided context does not have the corresponding schema
        for the provided reference schema.
        """

        class A(BaseModel):
            pass

        class B(BaseModel):
            x: A
            y: B

        class C(BaseModel):
            a: A
            c: C

        with pytest.raises(RuntimeError, match="not found in provided context"):
            resolve_ref_schema(C.__pydantic_core_schema__, B.__pydantic_core_schema__)


class TestGetFieldSchema:
    def test_valid_input(self):
        """
        Test with valid input
        """

        class A(BaseModel):
            a: int
            b: str

        class B(A):
            x: A
            y: B

        a_schema = A.__pydantic_core_schema__
        b_schema = B.__pydantic_core_schema__

        assert_err_msg = (
            "Wrong assumption about Pydantic behavior. Please re-write the test."
        )

        # If these two assertions fail, it doesn't mean `get_field_schema` is wrong.
        # It only means the assumption on how Pydantic represents models in `CoreSchema`
        # is wrong, and we have to find another way to test `get_field_schema`.
        assert a_schema["type"] == "model", assert_err_msg
        assert b_schema["type"] == "definitions", assert_err_msg

        a_field_schema_from_a = get_field_schema(A, "a")
        a_field_schema_from_b = get_field_schema(B, "a")
        assert (
            a_field_schema_from_a.schema
            == a_field_schema_from_b.schema
            == {"type": "int"}
        )
        assert a_field_schema_from_a.context == A.__pydantic_core_schema__
        assert a_field_schema_from_b.context == B.__pydantic_core_schema__
        assert a_field_schema_from_a.field_info is A.model_fields["a"]
        assert a_field_schema_from_b.field_info is B.model_fields["a"]

        b_field_schema_from_a = get_field_schema(A, "b")
        b_field_schema_from_b = get_field_schema(B, "b")
        assert (
            b_field_schema_from_a.schema
            == b_field_schema_from_b.schema
            == {"type": "str"}
        )
        assert b_field_schema_from_a.context == A.__pydantic_core_schema__
        assert b_field_schema_from_b.context == B.__pydantic_core_schema__
        assert b_field_schema_from_a.field_info is A.model_fields["b"]
        assert b_field_schema_from_b.field_info is B.model_fields["b"]

        # Verify the resolution of the field schema
        x_field_schema = get_field_schema(B, "x")
        assert x_field_schema.schema["type"] == "model"
        assert x_field_schema.schema["cls"] is A
        y_field_schema = get_field_schema(B, "y")
        assert y_field_schema.schema["type"] == "model"
        assert y_field_schema.schema["cls"] is B

    def test_input_without_model_fields(self):
        """
        Test input model without model fields
        """
        # noinspection PyPep8Naming
        Pets = RootModel[list[str]]

        with pytest.raises(
            NotImplementedError,
            match="This function currently doesn't support the inner schema of",
        ):
            get_field_schema(Pets, "root")


def test_get_locally_defined_fields():
    class A(BaseModel):
        a: str
        b: int
        c: ClassVar[str]

    class B(A):
        # Overriding definitions
        a: Optional[str]

        # New definitions
        x: float
        y: ClassVar[int]
        z: bool

    new, overriding = get_locally_defined_fields(B)

    assert set(new.keys()) == {"x", "z"}
    assert set(overriding.keys()) == {"a"}

    assert new["x"].schema == {"type": "float"}
    assert new["z"].schema == {"type": "bool"}

    assert overriding["a"].schema == {"type": "nullable", "schema": {"type": "str"}}


@pytest.mark.parametrize(
    "items, key_func, value_func, expected",
    [
        (
            list(range(10)),
            lambda x: "even" if x % 2 == 0 else "odd",
            None,
            {"even": list(range(0, 10, 2)), "odd": list(range(1, 10, 2))},
        ),
        (
            ("a", "abc", "bmz", "acd", "cad", "cba"),
            itemgetter(0),
            None,
            {"a": ["a", "abc", "acd"], "b": ["bmz"], "c": ["cad", "cba"]},
        ),
        (
            list(range(10)),
            lambda x: "even" if x % 2 == 0 else "odd",
            lambda x: x * 2,
            {"even": list(range(0, 20, 4)), "odd": list(range(2, 20, 4))},
        ),
        (
            ("a", "abc", "bmz", "acd", "cad", "cba"),
            itemgetter(0),
            lambda x: x[0],
            {"a": ["a", "a", "a"], "b": ["b"], "c": ["c", "c"]},
        ),
    ],
)
def test_bucketize(items, key_func, value_func, expected):
    assert bucketize(items, key_func, value_func) == expected


def test_ensure_unique_names():
    class A:
        pass

    class B(BaseModel):
        pass

    class C(Enum):
        C1 = auto()
        C2 = auto()

    class D(Enum):
        D1 = auto()
        D2 = auto()

    class Y:
        pass

    def func() -> list[Type]:
        """
        A internal function used to provide a separate namespace
        """

        class X:
            pass

        # noinspection PyShadowingNames
        class B:
            pass

        # noinspection PyShadowingNames
        class C(BaseModel):
            pass

        # noinspection PyShadowingNames
        class D(Enum):
            D3 = auto()
            D4 = auto()

        class Z(Enum):
            Z1 = auto()
            Z2 = auto()

        return [X, B, C, D, Z]

    local_clses = [A, B, C, D, Y]

    assert ensure_unique_names(*local_clses) is None
    assert ensure_unique_names(*func()) is None

    with pytest.raises(NameCollisionError) as exc_info:
        ensure_unique_names(*local_clses, *func())

    err_str = str(exc_info.value)

    # Assert three collision messages separated by semicolons
    assert err_str.count(";") == 2
    assert err_str.count("Name collision @ B: ") == 1
    assert err_str.count("Name collision @ C: ") == 1
    assert err_str.count("Name collision @ D: ") == 1


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("", ""),
        ("  ", ""),
        ("  a  ", "a"),
        ("a  b", "a b"),
        ("a b  c", "a b c"),
        ("a\nb", "a b"),
        ("a\n\nb", "a b"),
        ("a\n\n\nb", "a b"),
        ("\t ", ""),
        ("\t a \t \n b \t", "a b"),
    ],
)
def test_normalize_whitespace(input_str, expected):
    assert normalize_whitespace(input_str) == expected


def test_fetch_defs():
    from tests.assets import mock_module0, mock_module1

    models, enums = fetch_defs([mock_module0, mock_module1])

    assert models == {
        mock_module0.A,
        mock_module0.B,
        mock_module0.C,
        mock_module1.X,
        mock_module1.Y,
    }
    assert enums == {
        mock_module0.E0,
        mock_module0.E1,
        mock_module1.E2,
        mock_module1.E3,
        mock_module1.E4,
    }


class TestGetUuidRegex:
    @pytest.mark.parametrize(
        "version, expected_output",
        [
            (
                1,
                (
                    r"^(?:urn:uuid:)?"  # Optional "urn:uuid:" prefix
                    r"[0-9a-fA-F]{8}-?"  # 8 hex digits with optional hyphen
                    r"[0-9a-fA-F]{4}-?"  # 4 hex digits with optional hyphen
                    # Version and 3 hex digits with optional hyphen
                    r"1[0-9a-fA-F]{3}-?"
                    # Variant and 3 hex digits with optional hyphen
                    r"[89abAB][0-9a-fA-F]{3}-?"
                    r"[0-9a-fA-F]{12}$"  # 12 hex digits
                ),
            ),
            (
                4,
                (
                    r"^(?:urn:uuid:)?"  # Optional "urn:uuid:" prefix
                    r"[0-9a-fA-F]{8}-?"  # 8 hex digits with optional hyphen
                    r"[0-9a-fA-F]{4}-?"  # 4 hex digits with optional hyphen
                    # Version and 3 hex digits with optional hyphen
                    r"4[0-9a-fA-F]{3}-?"
                    # Variant and 3 hex digits with optional hyphen
                    r"[89abAB][0-9a-fA-F]{3}-?"
                    r"[0-9a-fA-F]{12}$"  # 12 hex digits
                ),
            ),
            (
                None,
                (
                    r"^(?:urn:uuid:)?"  # Optional "urn:uuid:" prefix
                    r"[0-9a-fA-F]{8}-?"  # 8 hex digits with optional hyphen
                    r"[0-9a-fA-F]{4}-?"  # 4 hex digits with optional hyphen
                    r"[0-9a-fA-F]{4}-?"  # 4 hex digits with optional hyphen
                    r"[0-9a-fA-F]{4}-?"  # 4 hex digits with optional hyphen
                    r"[0-9a-fA-F]{12}$"  # 12 hex digits
                ),
            ),
        ],
    )
    def test_valid_input(self, version, expected_output):
        assert get_uuid_regex(version) == expected_output

    @pytest.mark.parametrize("version", [0, 2, 6])
    def test_invalid_input(self, version):
        with pytest.raises(ValueError, match="Invalid UUID version"):
            get_uuid_regex(version)

    @pytest.mark.parametrize(
        "text, version, match_expected",
        [
            ("60c32af6-4b10-11ef-9ab2-0ecb4bcddcb5", 1, True),
            ("3f46ae03-c654-36b0-a55d-cd0aa042c9f2", 3, True),
            ("6b4c4599-1963-4d01-abbf-abdcb30ad9ff", 4, True),
            ("2cba86aa-e4d3-5340-9c8d-012bfe7d5d9d", 5, True),
            ("2cba86aa-e4d3-5340-9c8d-012bfe7d5d9d", None, True),
            # With some hyphens missing
            ("2cba86aae4d353409c8d012bfe7d5d9d", None, True),
            ("6b4c4599-19634d01-abbfabdcb30ad9ff", 4, True),
            # With mismatched version
            ("60c32af6-4b10-11ef-9ab2-0ecb4bcddcb5", 3, False),
            ("3f46ae03-c654-36b0-a55d-cd0aa042c9f2", 1, False),
            ("6b4c4599-1963-4d01-abbf-abdcb30ad9ff", 1, False),
            ("2cba86aa-e4d3-5340-9c8d-012bfe7d5d9d", 4, False),
            # With wrong variant
            ("60c32af6-4b10-11ef-0ab2-0ecb4bcddcb5", 1, False),
            ("3f46ae03-c654-36b0-755d-cd0aa042c9f2", 3, False),
            ("6b4c4599-1963-4d01-cbbf-abdcb30ad9ff", 4, False),
            ("2cba86aa-e4d3-5340-2c8d-012bfe7d5d9d", 5, False),
            # With wrong variant and version, though version doesn't really matter here
            # With some hyphens missing
            ("12345678123456781234567812345678", 4, False),
            # too long
            ("6b4c4599-1963-4d01-abbf-abdacb30ad9ff", 4, False),
            ("2cba86aae4d353409c8d012bfbe7d5d9d", None, False),
            # too short
            ("6b4c4599-19634d01-abbabdcb30ad9ff", 4, False),
            ("6b4c4599-1963-4d01-abf-abdcb30ad9ff", None, False),
            # too many consecutive hyphens
            ("3f46ae03-c654--36b0-a55d-cd0aa042c9f2", 3, False),
            # Arbitrary strings
            ("Hello world!", 4, False),
            ("Foobar", None, False),
        ],
    )
    @pytest.mark.parametrize("prepend_prefix", [True, False])
    def test_generated_regex_behavior(
        self, text, version, prepend_prefix, match_expected
    ):
        """
        Verify the behavior of the generated regex pattern
        """
        if prepend_prefix:
            text = f"urn:uuid:{text}"

        if match_expected:
            assert re.match(get_uuid_regex(version), text) is not None
        else:
            assert re.match(get_uuid_regex(version), text) is None


@pytest.mark.parametrize(
    "input_, expected_out",
    [
        (None, set()),
        ([1, 2, 3], {1, 2, 3}),
        ({1, 2, 3}, {1, 2, 3}),
        (set(), None),
        ({3, 4, 5}, None),
    ],
)
def test_force_to_set(input_, expected_out):
    if not isinstance(input_, set):
        assert force_to_set(input_) == expected_out
    else:
        assert force_to_set(input_) is input_
