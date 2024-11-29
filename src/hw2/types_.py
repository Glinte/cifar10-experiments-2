from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:  # https://github.com/patrick-kidger/jaxtyping/issues/70
    from torch import Tensor as TorchTensor
    from numpy import ndarray
    from jaxtyping import Array as JaxArray
    from tensorflow import TfTensor
    Array = Union[TorchTensor, ndarray, JaxArray, TfTensor]
else:
    arrays = []
    try:
        from torch import Tensor as TorchTensor
    except Exception:
        pass
    else:
        arrays.append(TorchTensor)
    try:
        from numpy import ndarray
    except Exception:
        pass
    else:
        arrays.append(ndarray)
    try:
        from jaxtyping import Array as JaxArray
    except Exception:
        pass
    else:
        arrays.append(JaxArray)
    try:
        from tensorflow import Tensor as TfTensor
    except Exception:
        pass
    else:
        arrays.append(TfTensor)
    Array = Union[tuple(arrays)]
