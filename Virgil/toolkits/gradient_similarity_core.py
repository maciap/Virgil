from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from typing import Union
from abc import ABC

from alibi.api.interfaces import Explainer
from alibi.api.interfaces import Explainer, Explanation
import copy
import warnings
from enum import Enum
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)
import numpy as np
from alibi.api.interfaces import Explainer, Explanation
# these functions we have  asym_dot, cos, dot
# _get_options_string we have 
from typing_extensions import Literal

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from typing import Any, Callable, List, Optional, Union


from importlib import import_module
from tqdm import tqdm 



if TYPE_CHECKING:
    import tensorflow
    import torch

DEFAULT_META_SIM: dict = {
    "name": None,
    "type": ["whitebox"],
    "explanations": ["local"],
    "params": {},
    "version": None
}


DEFAULT_DATA_SIM: dict = {
    "scores": None,
    "ordered_indices": None,
    "most_similar": None,
    "least_similar": None
}



    
class Task(str, Enum):
    """
    Enum of supported tasks.
    """
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

from enum import Enum


class Framework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


try:
    import tensorflow as tf  # noqa
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

try:
    import torch  # noqa
    has_pytorch = True
except ImportError:
    has_pytorch = False



class _PytorchBackend:
    device: Optional[torch.device] = None  # device used by `pytorch` backend

    @staticmethod
    def get_grads(
            model: nn.Module,
            X: Union[torch.Tensor, List[Any]],
            Y: torch.Tensor,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        """
        Computes the gradients of the loss function with respect to the model's parameters for a single training and
        target pair.

        Parameters
        ----------
        model
            The model to compute gradients for.
        X
            The input data.
        Y
            The target data.
        loss_fn
            The loss function to use.

        Returns
        -------
        grads
            The gradients of the loss function with respect to the model's parameters. This is returned as a flattened \
            array.
        """

        model.zero_grad()
        initial_model_state = model.training
        model.train(False)
        output = model(X)
        loss = loss_fn(output, Y)
        loss.backward()
        model.train(initial_model_state)

        return np.concatenate([_PytorchBackend._grad_to_numpy(grad=param.grad, name=name)
                               for name, param in model.named_parameters()
                               if param.grad is not None])

    @staticmethod
    def _grad_to_numpy(grad: torch.Tensor, name: Optional[str] = None) -> np.ndarray:
        """Convert gradient to `np.ndarray`.

        Converts gradient tensor to flat `numpy` array. If the gradient is a sparse tensor, it is converted to a dense
        tensor first.
        """
        if grad.is_sparse:
            grad = grad.to_dense()

        if not hasattr(grad, 'numpy'):
            name = f' for the named tensor: {name}' if name else ''
            raise TypeError((f'Could not convert gradient to `numpy` array{name}. To ignore these '
                             'gradients in the similarity computation set ``requires_grad=False`` on the '
                             'corresponding parameter.'))
        return grad.reshape(-1).cpu().numpy()

    @staticmethod
    def to_tensor(X: np.ndarray) -> torch.Tensor:
        """Converts a `numpy` array to a `pytorch` tensor and assigns to the backend device."""
        return torch.tensor(X).to(_PytorchBackend.device)

    @staticmethod
    def set_device(device: Union[str, int, torch.device, None] = None) -> None:
        """Sets the device to use for the backend.

        Allows the device used by the framework to be set using string, integer or device object directly. This is so
        users can follow the pattern recommended in
        https://pytorch.org/blog/pytorch-0_4_0-migration-guide/#writing-device-agnostic-code for writing
        device-agnostic code.
        """
        if isinstance(device, (int, str)):
            _PytorchBackend.device = torch.device(device)
        elif isinstance(device, torch.device):
            _PytorchBackend.device = device
        elif device is not None:
            raise TypeError(("`device` must be a ``None``, `string`, `integer` or "
                            f"`torch.device` object. Got {type(device)} instead."))

    @staticmethod
    def to_numpy(X: torch.Tensor) -> np.ndarray:
        """Maps a `pytorch` tensor to `np.ndarray`."""
        return X.detach().cpu().numpy()

    @staticmethod
    def argmax(X: torch.Tensor, dim=-1) -> torch.Tensor:
        """Returns the index of the maximum value in a tensor."""
        return torch.argmax(X, dim=dim)

    @staticmethod
    def _count_non_trainable(model: nn.Module) -> int:
        """Returns number of non trainable parameters.

        Returns the number of parameters that are non trainable. If no trainable parameter exists we raise
        a `ValueError`.
        """

        num_non_trainable_params = len([param for param in model.parameters() if not param.requires_grad])

        if num_non_trainable_params == len(list(model.parameters())):
            raise ValueError("The model has no trainable parameters. This method requires at least "
                             "one trainable parameter to compute the gradients for. "
                             "Try setting ``.requires_grad_(True)`` on the model or one of its parameters.")

        return num_non_trainable_params
    



def import_optional(module_name: str, names: Optional[List[str]] = None) -> Any:
    """Import a module that depends on optional dependencies

    Note: This function is used to import modules that depend on optional dependencies. Because it mirrors the python
    import functionality its return type has to be `Any`. Using objects imported with this function can lead to
    misspecification of types as `Any` when the developer intended to be more restrictive.

    Parameters
    ----------
    module_name
        The module to import
    names
        The names to import from the module. If None, all names are imported.

    Returns
    -------
    The module or named objects within the modules if names is not None. If the import fails due to a
    ModuleNotFoundError or ImportError then the requested module or named objects are replaced with instances of
    the MissingDependency class above.
    """

    try:
        module = import_module(module_name)
        # TODO: We should check against specific dependency versions here.
        if names is not None:
            objs = tuple(getattr(module, name) for name in names)
            return objs if len(objs) > 1 else objs[0]
        return module
    except (ImportError, ModuleNotFoundError) as err:
        if err.name is None:
            raise err
        name, *_ = err.name.split('.')
        if name not in ERROR_TYPES:
            raise err
        missing_dependency = ERROR_TYPES[name]
        if names is not None:
            missing_dependencies = \
                tuple(MissingDependency(
                    missing_dependency=missing_dependency,
                    object_name=name,
                    err=err) for name in names)
            return missing_dependencies if len(missing_dependencies) > 1 else missing_dependencies[0]
        return MissingDependency(
            missing_dependency=missing_dependency,
            object_name=module_name,
            err=err)
    
def _select_backend(backend: Framework = Framework.TENSORFLOW) \
        -> Union[Type['_TensorFlowBackend'], Type['_PytorchBackend']]:
    """
    Selects the backend according to the `backend` flag.

    Parameters
    ---------
    backend
        Deep learning backend.
    """
    # Check if pytorch/tensorflow backend supported.
    if (backend == Framework.PYTORCH and not has_pytorch) or \
            (backend == Framework.TENSORFLOW and not has_tensorflow):
        raise ImportError(f'{backend} not installed. Cannot initialize and run the GradientExplainer'
                          f' with {backend} backend.')

    # Allow only pytorch and tensorflow.
    elif backend not in [Framework.PYTORCH, Framework.TENSORFLOW]:
        raise NotImplementedError(f'{backend} not implemented. Use `tensorflow` or `pytorch` instead.')

    return _PytorchBackend




class BaseSimilarityExplainer(Explainer, ABC):
    """Base class for similarity explainers."""

    def __init__(self,
                 predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]''',
                 sim_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 precompute_grads: bool = False,
                 backend: Framework = Framework.TENSORFLOW,
                 device: 'Union[int, str, torch.device, None]' = None,
                 meta: Optional[dict] = None,
                 verbose: bool = False,
                 ):
        """Constructor

        Parameters
        ----------
        predictor
            Model to be explained.
        loss_fn
            Loss function.
        sim_fn
            Similarity function. Takes two inputs and returns a similarity value.
        precompute_grads
            Whether to precompute and store the gradients when fitting.
        backend
            Deep learning backend.
        device
            Device to be used. Will default to the same device the backend defaults to.
        meta
            Metadata specific to explainers that inherit from this class. Should be initialized in the child class and
            passed in here. Is used in the `__init__` of the base Explainer class.
        """

        # Select backend.
        self.backend = _select_backend(backend)
        self.backend.set_device(device)  # type: ignore

        self.predictor = predictor
        self.loss_fn = loss_fn
        self.sim_fn = sim_fn
        self.precompute_grads = precompute_grads
        self.verbose = verbose

        meta = {} if meta is None else meta
        super().__init__(meta=meta)

    def fit(self,
            X_train: Union[np.ndarray, List[Any]],
            Y_train: np.ndarray) -> "Explainer":
        """Fit the explainer. If ``self.precompute_grads == True`` then the gradients are precomputed and stored.

        Parameters
        ----------
        X_train
            Training data.
        Y_train
            Training labels.

        Returns
        -------
        self
            Returns self.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_dims = self.X_train.shape[1:] if isinstance(self.X_train, np.ndarray) else None
        self.Y_dims = self.Y_train.shape[1:]
        self.grad_X_train = np.array([])

        # compute and store gradients
        if self.precompute_grads:
            grads = []
            X: Union[np.ndarray, List[Any]]
            for X, Y in tqdm(zip(self.X_train, self.Y_train), disable=not self.verbose):
                grad_X_train = self._compute_grad(self._format(X), Y[None])
                grads.append(grad_X_train[None])

            self.grad_X_train = np.concatenate(grads, axis=0)
        return self

    @staticmethod
    def _is_tensor(x: Any) -> bool:
        _TfTensor = import_optional('tensorflow', ['Tensor'])
        _PtTensor = import_optional('torch', ['Tensor'])
        """Checks if an obejct is a tensor."""
        if has_tensorflow and isinstance(x, _TfTensor):
            return True
        if has_pytorch and isinstance(x, _PtTensor):
            return True
        if isinstance(x, np.ndarray):
            return True
        return False

    @staticmethod
    def _format(x: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any]'
                ) -> 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, List[Any]]':
        """Adds batch dimension."""
        if BaseSimilarityExplainer._is_tensor(x):
            return x[None]
        return [x]

    def _verify_fit(self) -> None:
        """Verify that the explainer has been fitted.

        Raises
        ------
        ValueError
            If the explainer has not been fitted.
        """
        if not hasattr(self, 'X_train') or not hasattr(self, 'Y_train'):
            raise ValueError('Training data not set. Call `fit` and pass training data first.')

    def _match_shape_to_data(self,
                             data: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]]',
                             target_type: Literal['X', 'Y']
                             ) -> 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, List[Any]]':
        """
        Verify the shape of `data` against the shape of the training data.

        Used to ensure input is correct shape for gradient methods implemented in the backends. `data` will be the
        features or label of the instance being explained. If the `data` is not a batch, reshape to be a single batch
        element. i.e. if training data shape is `(3, 28, 28)` and data shape is `(3, 28, 28)` we reshape data to
        `(1, 3, 28, 28)`.

        Parameters
        ----------
        data
            Data to be matched shape-wise against the training data.
        target_type
            Type of data: ``'X'`` | ``'Y'``. Used to determine if data should take the shape of predictor input or
            predictor output. ``'X'`` will utilize the `X_dims` attribute which stores the shape of the training data.
            ``'Y'`` will match the shape of `Y_dims` which is the shape of the target data.

        Raises
        ------
        ValueError
            If the shape of `data` does not match the shape of the training data, or fit has not been called prior to
            calling this method.
        """
        if self._is_tensor(data):
            return self._match_shape_to_data_tensor(data, target_type)
        return self._match_shape_to_data_any(data)

    def _match_shape_to_data_tensor(self,
                                    data: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
                                    target_type: Literal['X', 'Y']
                                    ) -> 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]':
        """ Verify the shape of `data` against the shape of the training data for tensor like data."""
        target_shape = getattr(self, f'{target_type}_dims')
        if data.shape == target_shape:
            data = data[None]
        if data.shape[1:] != target_shape:
            raise ValueError((f'Input `{target_type}` has shape {data.shape[1:]}'
                              f' but training data has shape {target_shape}'))
        return data

    @staticmethod
    def _match_shape_to_data_any(data: Union[Any, List[Any]]) -> list:
        """ Ensures that any other data type is a list."""
        if isinstance(data, list):
            return data
        return [data]

    def _compute_adhoc_similarity(self, grad_X: np.ndarray) -> np.ndarray:
        """
        Computes the similarity between the gradients of the test instances and all the training instances. The method
        performs the computation of the gradients of the training instance on the fly without storing them in memory.

        parameters
        ----------
        grad_X
            Gradients of the test instances.
        """
        scores = np.zeros((len(grad_X), len(self.X_train)))
        X: Union[np.ndarray, List[Any]]
        for i, (X, Y) in tqdm(enumerate(zip(self.X_train, self.Y_train)), disable=not self.verbose):
            grad_X_train = self._compute_grad(self._format(X), Y[None])
            scores[:, i] = self.sim_fn(grad_X, grad_X_train[None])[:, 0]
        return scores

    def _compute_grad(self,
                      X: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, List[Any]]',
                      Y: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]') \
            -> np.ndarray:
        """Computes predictor parameter gradients and returns a flattened `numpy` array."""
        X = self.backend.to_tensor(X) if isinstance(X, np.ndarray) else X
        Y = self.backend.to_tensor(Y) if isinstance(Y, np.ndarray) else Y
        return self.backend.get_grads(self.predictor, X, Y, self.loss_fn)

    def reset_predictor(self, predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]') -> None:
        """Resets the predictor to the given predictor.

        Parameters
        ----------
        predictor
            The new predictor to use.
        """
        self.predictor = predictor





def dot(X: np.ndarray, Y: np.ndarray) -> Union[float, np.ndarray]:
    """
    Performs a dot product between the vector(s) in X and vector Y. (:math:`X^T Y = \\sum_i X_i Y_i`). Each of `X` and
    `Y` should have a leading batch dimension of size at least 1.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Matrix of vectors.

    Returns
    -------
        Matrix of dot products between the vector(s) in X and vectors in Y.
    """
    assert len(X.shape) > 1 and len(Y.shape) > 1, "The vectors `X` and `Y` should have a leading batch dimension."
    assert X.shape[1] == Y.shape[1], "The second dimension of `X` needs to be the same as the dimension of `Y`."
    return np.dot(X, Y.T)


def cos(X: np.ndarray, Y: np.ndarray, eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Computes the cosine between the vector(s) in X and vector Y. (:math:`X^T Y/\\|X\\|\\|Y\\|`). Each of `X` and `Y`
    should have a leading batch dimension of size at least 1.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Matrix of vectors.
    eps
        Numerical stability.

    Returns
    -------
        Matrix of cosine similarities between the vector(s) in X and vectors in Y.
    """
    assert len(X.shape) > 1 and len(Y.shape) > 1, "The vectors `X` and `Y` should have a leading batch dimension."
    assert X.shape[1] == Y.shape[1], "The second dimension of `X` needs to be the same as the dimension of `Y`."
    denominator = np.linalg.norm(X, axis=1)[:, None] @ np.linalg.norm(Y, axis=1)[None, :]
    return np.dot(X, Y.T) / (denominator + eps)


def asym_dot(X: np.ndarray, Y: np.ndarray, eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Computes the influence of training instances `Y` to test instances `X`. This is an asymmetric kernel.
    (:math:`X^T Y/\\|Y\\|^2`). See the `paper <https://arxiv.org/abs/2102.05262>`_ for more details. Each of `X` and
    `Y` should have a leading batch dimension of size at least 1.

    Parameters
    ----------
    X
        Matrix of vectors.
    Y
        Matrix of vectors.
    eps
        Numerical stability.

    Returns
    -------
        Matrix of asymmetric dot product similarity values between the vector(s) in X and vectors in Y.
    """

    assert len(X.shape) > 1 and len(Y.shape) > 1, "The vectors `X` and `Y` should have a leading batch dimension."
    assert X.shape[1] == Y.shape[1], "The second dimension of `X` needs to be the same as the dimension of `Y`."
    denominator = np.linalg.norm(Y, axis=1) ** 2
    return np.dot(X, Y.T) / (denominator + eps)[None, :]

def _get_options_string(enum: Type[Enum]) -> str:
    """Get the enums options seperated by pipe as a string.
    Note: this only works on enums inheriting from `str`, i.e. class MyEnum(str, Enum).
    Note: Python 3.11 will introduce enum.StrEnum which will be the preferred type for string enumerations.
    If we want finer control over typing we could define a new type."""
    return f"""'{"' | '".join(enum)}'"""





class GradientSimilarity(BaseSimilarityExplainer):

    def __init__(self,
                 predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]''',
                 sim_fn: Literal['grad_dot', 'grad_cos', 'grad_asym_dot'] = 'grad_dot',
                 task: Literal['classification', 'regression'] = 'classification',
                 precompute_grads: bool = False,
                 backend: Literal['tensorflow', 'pytorch'] = 'tensorflow',
                 device: 'Union[int, str, torch.device, None]' = None,
                 verbose: bool = False,
                 ):
        """`GradientSimilarity` explainer.

        The gradient similarity explainer is used to find examples in the training data that the predictor considers
        similar to test instances the user wants to explain. It uses the gradients of the loss between the model output
        and the training data labels. These are compared using the similarity function specified by ``sim_fn``. The
        `GradientSimilarity` explainer can be applied to models trained for both classification and regression tasks.


        Parameters
        ----------
        predictor
            Model to explain.
        loss_fn
            Loss function used. The gradient of the loss function is used to compute the similarity between the test
            instances and the training set.
        sim_fn
            Similarity function to use. The ``'grad_dot'`` similarity function computes the dot product of the
            gradients, see :py:func:`alibi.explainers.similarity.metrics.dot`. The ``'grad_cos'`` similarity function
            computes the cosine similarity between the gradients, see
            :py:func:`alibi.explainers.similarity.metrics.cos`. The ``'grad_asym_dot'`` similarity function is similar
            to ``'grad_dot'`` but is asymmetric, see :py:func:`alibi.explainers.similarity.metrics.asym_dot`.
        task
            Type of task performed by the model. If the task is ``'classification'``, the target value passed to the
            explain method of the test instance can be specified either directly or left  as ``None``, if left ``None``
            we use the model's maximum prediction. If the task is ``'regression'``, the target value of the test
            instance must be specified directly.
        precompute_grads
            Whether to precompute the gradients. If ``False``, gradients are computed on the fly otherwise we
            precompute them which can be faster when it comes to computing explanations. Note this option may be memory
            intensive if the model is large.
        backend
            Backend to use.
        device
            Device to use. If ``None``, the default device for the backend is used. If using `pytorch` backend see
            `pytorch device docs <https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>`_ for correct
            options. Note that in the `pytorch` backend case this parameter can be a ``torch.device``. If using
            `tensorflow` backend see `tensorflow docs <https://www.tensorflow.org/api_docs/python/tf/device>`_ for
            correct options.
        verbose
            Whether to print the progress of the explainer.

        Raises
        ------
        ValueError
            If the ``task`` is not ``'classification'`` or ``'regression'``.
        ValueError
            If the ``sim_fn`` is not ``'grad_dot'``, ``'grad_cos'`` or ``'grad_asym_dot'``.
        ValueError
            If the ``backend`` is not ``'tensorflow'`` or ``'pytorch'``.
        TypeError
            If the device is not an ``int``, ``str``, ``torch.device`` or ``None`` for the torch backend option or if
            the device is not ``str`` or ``None`` for the tensorflow backend option.
        """
        # TODO: add link to docs page for GradientSimilarity explainer in the docstring once written

        sim_fn_opts: Dict[str, Callable] = {
            'grad_dot': dot,
            'grad_cos': cos,
            'grad_asym_dot': asym_dot
        }

        if sim_fn not in sim_fn_opts.keys():
            raise ValueError(f"""Unknown method {sim_fn}. Consider using: '{"' | '".join(sim_fn_opts.keys())}'.""")

        resolved_sim_fn = sim_fn_opts[sim_fn]

        if task not in Task.__members__.values():
            raise ValueError(f"Unknown task {task}. Consider using: {_get_options_string(Task)}.")

        self.task = task

        if backend not in Framework.__members__.values():
            raise ValueError(f"Unknown backend {backend}. Consider using: {_get_options_string(Framework)}.")

        super().__init__(predictor, loss_fn, resolved_sim_fn, precompute_grads, Framework(backend), device=device,
                         meta=copy.deepcopy(DEFAULT_META_SIM), verbose=verbose)

        self.meta['params'].update(
            sim_fn_name=sim_fn,
            store_grads=precompute_grads,
            backend_name=backend,
            task_name=task
        )

        num_non_trainable = self.backend._count_non_trainable(self.predictor)
        if num_non_trainable:
            warning_msg = (f"Found {num_non_trainable} non-trainable parameters in the model. These parameters "
                           "don't have gradients and will not be included in the computation of gradient similarity."
                           " This might be because your model has layers that track statistics using non-trainable "
                           "parameters such as batch normalization layers. In this case, you don't need to worry. "
                           "Otherwise it's because you have set some parameters to be non-trainable and alibi is "
                           "letting you know.")
            warnings.warn(warning_msg)

    def fit(self,
            X_train: Union[np.ndarray, List[Any]],
            Y_train: np.ndarray) -> "Explainer":
        """Fit the explainer.

        The `GradientSimilarity` explainer requires the model gradients over the training data. In the explain method
        it compares them to the model gradients for the test instance(s). If ``precompute_grads=True`` on
        initialization then the gradients are precomputed here and stored. This will speed up the explain method call
        but storing the gradients may not be feasible for large models.

        Parameters
        ----------
        X_train
            Training data.
        Y_train
            Training labels.

        Returns
        -------
        self
            Returns self.
        """
        return super().fit(X_train, Y_train)

    def _preprocess_args(
            self,
            X: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]]',
            Y: 'Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]]' = None) \
            -> 'Union[Tuple[torch.Tensor, torch.Tensor], Tuple[tensorflow.Tensor, tensorflow.Tensor]]':
        """Formats `X`, `Y` for explain method.

        Parameters
        ----------
        X
            Input data requiring formatting.
        Y
            Target data requiring formatting.

        Returns
        -------
        X
            Input data formatted for explain method.
        Y
            Target data formatted for explain method.

        """
        X = self._match_shape_to_data(X, 'X')
        if isinstance(X, np.ndarray):
            X = self.backend.to_tensor(X)

        if self.task == Task.REGRESSION and Y is None:
            err_msg = "Regression task requires a target value. 'Y' must be provided."
            raise ValueError(err_msg)

        if Y is None:
            Y = self.predictor(X)
            Y = self.backend.argmax(Y)  # type: ignore

        Y = self._match_shape_to_data(Y, 'Y')
        if isinstance(Y, np.ndarray):
            Y = self.backend.to_tensor(Y)

        return X, Y

    def explain(
            self,
            X: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]]',
            Y: 'Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]]' = None) -> "Explanation":
        """Explain the predictor's predictions for a given input.

        Computes the similarity score between the inputs and the training set. Returns an explainer object
        containing the scores, the indices of the training set instances sorted by descending similarity and the
        most similar and least similar instances of the data set for the input. Note that the input may be a single
        instance or a batch of instances.

        Parameters
        ----------
        X
            `X` can be a `numpy` array, `tensorflow` tensor, `pytorch` tensor of the same shape as the training data
            or a list of objects, with or without a leading batch dimension. If the batch dimension is missing it's
            added.
        Y
            `Y` can be a `numpy` array, `tensorflow` tensor or a `pytorch` tensor. In the case of a regression task, the
            `Y` argument must be present. If the task is classification then `Y` defaults to the model prediction.

        Returns
        -------
        `Explanation` object containing the ordered similarity scores for the test instance(s) with additional \
        metadata as attributes. Contains the following data-related attributes
            - `scores`: ``np.ndarray`` - similarity scores for each pair of instances in the training and test set \
            sorted in descending order.
            - `ordered_indices`: ``np.ndarray`` - indices of the paired training and test set instances sorted by the \
            similarity score in descending order.
            - `most_similar`: ``np.ndarray`` - 5 most similar instances in the training set for each test instance \
            The first element is the most similar instance.
            -  `least_similar`: ``np.ndarray`` - 5 least similar instances in the training set for each test instance. \
            The first element is the least similar instance.

        Raises
        ------
        ValueError
            If `Y` is ``None`` and the `task` is ``'regression'``.
        ValueError
            If the shape of `X` or `Y` does not match the shape of the training or target data.
        ValueError
            If the fit method has not been called prior to calling this method.
        """
        self._verify_fit()
        X, Y = self._preprocess_args(X, Y)
        test_grads = []
        for x, y in zip(X, Y):
            test_grads.append(self._compute_grad(self._format(x), y[None])[None])
        grads_X_test = np.concatenate(np.array(test_grads), axis=0)
        if not self.precompute_grads:
            scores = self._compute_adhoc_similarity(grads_X_test)
        else:
            scores = self.sim_fn(grads_X_test, self.grad_X_train)
        return self._build_explanation(scores)

    def _build_explanation(self, scores: np.ndarray) -> "Explanation":
        """Builds an explanation object.

        Parameters
        ----------
        scores
            The scores for each of the instances in the data set computed by the similarity method.
        """
        data = copy.deepcopy(DEFAULT_DATA_SIM)
        sorted_score_indices = np.argsort(scores)[:, ::-1]
        most_similar: Union[np.ndarray, List[Any]]
        least_similar: Union[np.ndarray, List[Any]]

        if isinstance(self.X_train, np.ndarray):
            broadcast_indices = np.expand_dims(
                sorted_score_indices,
                axis=tuple(range(2, len(self.X_train[None].shape)))
            )
            most_similar = np.take_along_axis(self.X_train[None], broadcast_indices[:, :5], axis=1)
            least_similar = np.take_along_axis(self.X_train[None], broadcast_indices[:, -1:-6:-1], axis=1)
        else:
            most_similar = [[self.X_train[i] for i in ssi[:5]] for ssi in sorted_score_indices]
            least_similar = [[self.X_train[i] for i in ssi[-1:-6:-1]] for ssi in sorted_score_indices]

        data.update(
            scores=np.take_along_axis(scores, sorted_score_indices, axis=1),
            ordered_indices=sorted_score_indices,
            most_similar=most_similar,
            least_similar=least_similar
        )
        return Explanation(meta=self.meta, data=data)

        
    

