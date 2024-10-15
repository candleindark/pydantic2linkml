from enum import Enum

from pydantic import BaseModel, RootModel


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
