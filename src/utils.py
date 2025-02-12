"""A protocol that all policies should follow.

This provides a mechanism for type-hinting and isinstance checks without requiring the policies classes
subclass a base class.

The protocol structure, method signatures, and docstrings should be used by developers as a reference for
how to implement new policies.
"""

from typing import Optional, Protocol, runtime_checkable
from mindspore import Tensor

from transformers import AutoTokenizer


@runtime_checkable
class PolicyModel(Protocol):
    """The required interface for implementing a policy.

    We also expect all policies to subclass mindspore.nn.Cell
    """

    name: str

    def generate(self, *args, **kwargs):
        """To be called when generating results"""

    def logits(self, input_ids: Tensor, num_logits_to_keep: Optional[int] = None) -> Tensor:
        """get logits from models"""



