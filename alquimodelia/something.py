
from alquimodelia.models_definitions import MODELS_ARCHS, get_model_from_def

# os.environ["KERAS_BACKEND"] = "torch"  # @param ["tensorflow", "jax", "torch"]


input_args = {
    "X_timeseries": 6, # Number of sentinel images
    "Y_timeseries": 1, # Number of volume maps
    "n_features_train": 12, # Number of sentinel bands
    "n_features_predict": 1, # We just want to predict the volume linearly
}
input_args = {
    "X_timeseries": 168, # Number of sentinel images
    "Y_timeseries": 24, # Number of volume maps
    "n_features_train": 18, # Number of sentinel bands
    "n_features_predict": 1, # We just want to predict the volume linearly
}
UNET = get_model_from_def("UNET", input_args=input_args,model_structures=MODELS_ARCHS,)
TRANSFORMER = get_model_from_def("Transformer", input_args=input_args,model_structures=MODELS_ARCHS,)
Stacked6Transformer = get_model_from_def("Stacked6Transformer", input_args=input_args,model_structures=MODELS_ARCHS,)

StackedCNN = get_model_from_def("StackedCNN", input_args=input_args,model_structures=MODELS_ARCHS,)

UNET.summary()
TRANSFORMER.summary()
StackedCNN.summary()

StackedDense = get_model_from_def("StackedDense", input_args=input_args,model_structures=MODELS_ARCHS,)
StackedDense.summary()

import numpy as np

# Create the test data
input_shape = (10912, *TRANSFORMER.input_shape[1:])
output_shape = (10912, *TRANSFORMER.output_shape[1:])
np.prod(input_shape)
X_test = np.random.rand(np.prod(input_shape)).reshape(input_shape)
Y_test = np.random.rand(np.prod(output_shape)).reshape(output_shape)

compile_args = {
    "optimizer": "adam",
    "loss": "mse",
}
TRANSFORMER.compile(**compile_args)
TRANSFORMER.fit(X_test, Y_test, epochs=2)

StackedDense.compile(**compile_args)
StackedDense.fit(X_test, Y_test, epochs=2)

Stacked6Transformer.compile(**compile_args)
Stacked6Transformer.fit(X_test, Y_test, epochs=2)