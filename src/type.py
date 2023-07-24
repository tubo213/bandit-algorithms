from typing import Literal, Union

from src.policy.default.base import AbstractContextFreePolicy, AbstractLinearPolicy
from src.policy.multiple_play.base import (
    AbstractMultiplePlayContextFreePolicy,
    AbstractMultiplePlayLinearPolicy,
)

POLICY_TYPE = Union[AbstractContextFreePolicy, AbstractLinearPolicy]
MUTIPLE_PLAY_POLICY_TYPE = Union[
    AbstractMultiplePlayContextFreePolicy, AbstractMultiplePlayLinearPolicy
]
TASK_TYPES = Literal["regression", "binary"]
