from __future__ import annotations

from typing import cast

import pytest


def test_get_parent_model():
    from pydantic2linkml.tools import get_parent_model
    from pydantic import BaseModel

    class Foo:
        pass

    class Bar:
        pass

    class A(BaseModel):
        pass

    class B(Foo, A):
        pass

    class C(B, Bar):
        pass

    class X(BaseModel):
        pass

    class Z(X, C):
        pass

    assert get_parent_model(BaseModel) is None
    assert get_parent_model(A) is BaseModel
    assert get_parent_model(B) is A
    assert get_parent_model(C) is B

    with pytest.raises(ValueError, match="multiple Pydantic base models"):
        get_parent_model(Z)


class TestResolveRefSchema:
    def test_valid_input(self):
        """
        Test with valid input
        """
        from typing import Optional

        from pydantic2linkml.tools import resolve_ref_schema
        from pydantic import BaseModel
        from pydantic_core import core_schema

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
        # It only means the assumption about how Pydantic uses `definition-ref` is wrong,
        # and we have to find another way to test `resolve_ref_schema`.
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
        from pydantic import BaseModel
        from pydantic2linkml.tools import resolve_ref_schema

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
        from pydantic import BaseModel
        from pydantic2linkml.tools import resolve_ref_schema

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
        from pydantic2linkml.tools import get_field_schema
        from pydantic import BaseModel

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
        assert a_field_schema_from_a == a_field_schema_from_b == {"type": "int"}

        b_field_schema_from_a = get_field_schema(A, "b")
        b_field_schema_from_b = get_field_schema(B, "b")
        assert b_field_schema_from_a == b_field_schema_from_b == {"type": "str"}

        x_field_schema = get_field_schema(B, "x")
        assert x_field_schema["type"] == "model"
        assert x_field_schema["cls"] is A

        y_field_schema = get_field_schema(B, "y")
        assert y_field_schema["type"] == "model"
        assert y_field_schema["cls"] is B

    def test_input_without_model_fields(self):
        """
        Test input model without model fields
        """
        from pydantic2linkml.tools import get_field_schema
        from pydantic import RootModel

        # noinspection PyPep8Naming
        Pets = RootModel[list[str]]

        with pytest.raises(
            NotImplementedError,
            match="This function currently doesn't support the inner schema of",
        ):
            get_field_schema(Pets, "dummy")


def test_get_locally_defined_fields():
    from pydantic2linkml.tools import get_locally_defined_fields
    from pydantic import BaseModel

    from typing import ClassVar, Optional

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
