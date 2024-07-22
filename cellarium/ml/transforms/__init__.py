# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from cellarium.ml.transforms.divide_by_scale import DivideByScale
from cellarium.ml.transforms.filter import Filter
from cellarium.ml.transforms.log1p import Log1p
from cellarium.ml.transforms.normalize_total import NormalizeTotal
from cellarium.ml.transforms.z_score import ZScore
from cellarium.ml.core.pipeline import CellariumPipeline
from torch import nn
from functools import wraps


def before_to_device(transform: nn.Module) -> nn.Module:
    """
    Wrapper that makes a generic :class:`~torch.nn.Module` transform into a 
    class:`~cellarium.ml.transforms.BeforeBatchTransferTransform`.

    Functionally, class:`~cellarium.ml.transforms.BeforeBatchTransferTransform`s are 
    applied before the batch is loaded on the device during model training. This is 
    useful for things like filtering, where we do not need to load the entire batch on the device.

    In implementation, the list of applied transforms can be restricted to those which are or are not 
    subclasses of class:`~cellarium.ml.transforms.BeforeBatchTransferTransform` depending on the context.
    This allows us to call a module's ``self.pipeline`` and have it work as expected, whether or not it is 
    being invoked inside a training loop where the ``on_before_batch_transfer`` hook is called.
    """
    return BeforeBatchTransferTransform(transform)


class BeforeBatchTransferTransform(nn.Module):
    """Transform wrapper used only as an indicator for :class:`~cellarium.ml.transforms.BatchTransferContext`"""
    def __init__(self, transform: nn.Module):
        super().__init__()
        self.transform_name = transform.__repr__()
        self.transform = transform
        self.forward = self.transform.forward

    def __repr__(self) -> str:
        return f"before_to_device({self.transform_name})"


class BeforeBatchTransferContext:
    """Replace a pipeline by the transforms that are instances of 
    :class:`~cellarium.ml.transforms.BeforeBatchTransferTransform`"""
    def __init__(self, module: "CellariumModule"):
        self.module = module

    def __enter__(self):
        self.original_pipeline = self.module.pipeline
        transforms = self.original_pipeline[:-1]
        self.module.pipeline = CellariumPipeline([t for t in transforms if isinstance(t, BeforeBatchTransferTransform)])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module.pipeline = self.original_pipeline
        del self.original_pipeline
        return False
    

class AfterBatchTransferContext:
    """Replace a pipeline by the transforms that are not instances of 
    :class:`~cellarium.ml.transforms.BeforeBatchTransferTransform`, plus the model"""
    def __init__(self, module: "CellariumModule"):
        self.module = module

    def __enter__(self):
        self.original_pipeline = self.module.pipeline
        transforms = self.original_pipeline[:-1]
        model = self.original_pipeline[-1]
        self.module.pipeline = CellariumPipeline([t for t in transforms if not isinstance(t, BeforeBatchTransferTransform)] + [model])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module.pipeline = self.original_pipeline
        del self.original_pipeline
        return False
    

def before_batch_transfer(method):
    """Decorator to wrap a method with a context manager that replaces the pipeline with the pre-batch-transfer transforms"""
    @wraps(method)
    def wrapper(obj, *args, **kwargs):
        with BeforeBatchTransferContext(obj):
            return method(obj, *args, **kwargs)
    return wrapper


def after_batch_transfer(method):
    """Decorator to wrap a method with a context manager that replaces the pipeline with the post-batch-transfer pipeline"""
    @wraps(method)
    def wrapper(obj, *args, **kwargs):
        with AfterBatchTransferContext(obj):
            return method(obj, *args, **kwargs)
    return wrapper


__all__ = ["DivideByScale", "Filter", "Log1p", "NormalizeTotal", "ZScore", 
           "before_to_device", "before_batch_transfer", "after_batch_transfer"]
