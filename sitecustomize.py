"""Compatibility shims for keras-vggface on Keras 3.

This module is imported automatically by Python when it is present on the
import path. It restores a few legacy Keras 2 modules that keras-vggface still
imports, while delegating the actual implementations to Keras 3 where possible.
"""

from types import ModuleType
import sys


def _install_module(module_name):
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    return module


try:
    import keras
except Exception:
    keras = None


if keras is not None:
    def _sanitize_keras_name(name):
        if isinstance(name, str):
            return name.replace("/", "_")
        return name

    from keras.src.layers import layer as keras_layer_module
    from keras.src.ops import operation as keras_operation_module

    original_layer_init = keras_layer_module.Layer.__init__

    def patched_layer_init(self, *args, **kwargs):
        if "name" in kwargs:
            kwargs["name"] = _sanitize_keras_name(kwargs["name"])
        return original_layer_init(self, *args, **kwargs)

    keras_layer_module.Layer.__init__ = patched_layer_init

    original_operation_init = keras_operation_module.Operation.__init__

    def patched_operation_init(self, name=None):
        return original_operation_init(self, name=_sanitize_keras_name(name))

    keras_operation_module.Operation.__init__ = patched_operation_init

    if "keras.utils.data_utils" not in sys.modules:
        data_utils = _install_module("keras.utils.data_utils")
        data_utils.get_file = keras.utils.get_file
        keras.utils.data_utils = data_utils

    if "keras.utils.layer_utils" not in sys.modules:
        layer_utils = _install_module("keras.utils.layer_utils")

        def convert_all_kernels_in_model(model):
            return model

        def convert_dense_weights_data_format(dense, shape, data_format):
            return dense

        layer_utils.convert_all_kernels_in_model = convert_all_kernels_in_model
        layer_utils.convert_dense_weights_data_format = convert_dense_weights_data_format
        keras.utils.layer_utils = layer_utils

    if "keras.engine" not in sys.modules:
        engine = _install_module("keras.engine")
        engine.__path__ = []
        keras.engine = engine

    topology = _install_module("keras.engine.topology")
    topology.get_source_inputs = keras.utils.get_source_inputs
    sys.modules["keras.engine"].topology = topology