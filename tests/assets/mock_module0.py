from pydantic import BaseModel, RootModel
from enum import Enum


class A(BaseModel):
    pass


class B(BaseModel):
    pass


class C(A):
    pass


class E0(Enum):
    pass


class E1(Enum):
    pass


class R(RootModel):
    root: list[str]
