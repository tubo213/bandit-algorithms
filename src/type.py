from typing import Union

from src.policy.base import AbstractContextFreePolicy, AbstractLinearPolicy

POLICY_TYPE = Union[AbstractContextFreePolicy, AbstractLinearPolicy]
