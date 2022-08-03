from captum.log import log_usage
from captum._utils.common import _select_targets
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable

from .base import _base_metric


def _sufficiency(
    prob_original: Tensor, prob_pert: Tensor, target: Tensor
) -> Tensor:
    return _select_targets(prob_original, target) - _select_targets(
        prob_pert, target
    )


@log_usage()
def sufficiency(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    topk: float = 0.2,
) -> Tensor:
    # Inverse topk to select the non topk
    topk = 1.0 - topk

    return _base_metric(
        metric=_sufficiency,
        forward_func=forward_func,
        inputs=inputs,
        attributions=attributions,
        baselines=baselines,
        additional_forward_args=additional_forward_args,
        target=target,
        topk=topk,
        largest=False,
    )
