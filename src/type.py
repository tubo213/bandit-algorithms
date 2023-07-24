from typing import Literal, Union

from src.policy.base import AbstractContextFreePolicy, AbstractLinearPolicy

POLICY_TYPE = Union[AbstractContextFreePolicy, AbstractLinearPolicy]
TASK_TYPES = Literal["regression", "binary"]
