from alquimodelia.models_definitions import MODELS_ARCHS, get_model_from_def

# os.environ["KERAS_BACKEND"] = "torch"  # @param ["tensorflow", "jax", "torch"]
"c:/Users/e709729/Documents/tensorflow_datasets/74.jpg"
input_args = {
    "X_timeseries": 6,  # Number of sentinel images
    "Y_timeseries": 1,  # Number of volume maps
    "n_features_train": 12,  # Number of sentinel bands
    "n_features_predict": 1,  # We just want to predict the volume linearly
}
input_args = {
    "X_timeseries": 168,  # Number of sentinel images
    "Y_timeseries": 24,  # Number of volume maps
    "n_features_train": 18,  # Number of sentinel bands
    "n_features_predict": 1,  # We just want to predict the volume linearly
}
UNET = get_model_from_def(
    "UNET",
    input_args=input_args,
    model_structures=MODELS_ARCHS,
)
TRANSFORMER = get_model_from_def(
    "Transformer",
    input_args=input_args,
    model_structures=MODELS_ARCHS,
)
Stacked6Transformer = get_model_from_def(
    "Stacked6Transformer",
    input_args=input_args,
    model_structures=MODELS_ARCHS,
)

StackedCNN = get_model_from_def(
    "StackedCNN",
    input_args=input_args,
    model_structures=MODELS_ARCHS,
)

UNET.summary()
TRANSFORMER.summary()
StackedCNN.summary()

StackedDense = get_model_from_def(
    "StackedDense",
    input_args=input_args,
    model_structures=MODELS_ARCHS,
)
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


model_UNET = UNet(
    height=128,
    width=128,
    num_features_to_train=12,
    num_classes=1,
    x_timesteps=8,
    y_timesteps=1,
    activation_end="relu",
)


model_UNET_res = UNet(
    height=128,
    width=128,
    num_features_to_train=12,
    num_classes=1,
    x_timesteps=8,
    y_timesteps=1,
    residual=True,
    activation_end="relu",
)


model_UNET_att = UNet(
    height=128,
    width=128,
    num_features_to_train=12,
    num_classes=1,
    x_timesteps=8,
    y_timesteps=1,
    attention=True,
    residual=True,
    activation_end="relu",
)

in_sha = list(model_UNET_att.model.input_shape)
in_sha[0] = 2
in_sha = tuple(in_sha)
out_sha = list(model_UNET_att.model.output_shape)
out_sha[0] = 2
out_sha = tuple(out_sha)

X = np.random.rand(np.prod(in_sha)).reshape(in_sha)
Y = np.random.rand(np.prod(out_sha)).reshape(out_sha)

model_UNET_att.model.compile(loss="mse", optimizer="adam")
model_UNET_att.model.fit(X, Y, epochs=10)
