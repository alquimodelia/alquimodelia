import keras
from keras.models import Model


class ModelMagia:
    def model_input_shape(self):
        raise NotImplementedError

    def define_input_layer(self):
        self.input_layer = keras.Input(self.model_input_shape)

    def define_output_layer(self):
        raise NotImplementedError

    def get_last_layer_activation(self):
        raise NotImplementedError

    def model_setup(self):
        raise NotImplementedError

    def __init__(
        self,
        timesteps: int = 1,
        height: int = 1,
        width: int = 1,
        num_features_to_train: int = 1,
        num_classes: int = 1,
        x_timesteps: int = None,
        x_height: int = None,
        x_width: int = None,
        y_timesteps: int = None,
        y_height: int = None,
        y_width: int = None,
        activation_final: str = "sigmoid",
        data_format: str = "channels_last",
        **kwargs,
    ):
        # shape (N, T, H, W, C)
        self.x_timesteps = x_timesteps or timesteps
        self.x_height = x_height or height
        self.x_width = x_width or width
        self.num_features_to_train = num_features_to_train  # channels
        self.input_dimensions = (self.x_timesteps, self.x_height, self.x_width)
        self.input_dimensions_channels = (
            *self.input_dimensions,
            self.num_features_to_train,
        )

        self.y_timesteps = y_timesteps or timesteps
        self.y_height = y_height or height
        self.y_width = y_width or width
        self.num_classes = num_classes
        self.output_dimensions = (
            self.y_timesteps,
            self.y_height,
            self.y_width,
        )

        self.activation_final = activation_final
        self.data_format = data_format
        if self.data_format == "channels_first":
            self.channels_dimension = 0
        elif self.data_format == "channels_last":
            self.channels_dimension = -1

        self.model_setup()
        self.define_input_layer()
        self.define_output_layer()
        self.model = Model(
            inputs=self.input_layer, outputs=self.output_layer, **kwargs
        )
