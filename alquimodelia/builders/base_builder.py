import keras
from keras.models import Model


class BaseBuilder:
    # Define a self so subclasses can overwrite with inputs having other names (TinEye bands -> num_features_to_train)
    num_features_to_train = None
    num_classes = None

    def model_input_shape(self):
        raise NotImplementedError

    def define_input_layer(self):
        # This is the base for an input layer. It can be overwrite, but it should set this variable
        self.input_layer = keras.Input(self.model_input_shape)

    def get_input_layer(self):
        # This is to get the layer to enter the arch, here you can add augmentation or other processing
        return self.input_layer

    def define_model(self):
        # This is the model definition and it must return the output_layer of the architeture. which can be modified further
        # It should use the get_input_layer to fetch the inital layer.
        # it should set the self.last_arch_layer
        raise NotImplementedError

    def define_output_layer(self):
        # This should only deal with the last layer, it can be used to define the classification for instance, or multiple methods to close the model
        # it should use the last_arch_layer
        # it should define self.output_layer
        raise NotImplementedError

    def model_setup(self):
        # Any needed setup before building and conecting the layers
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
        self.num_features_to_train = (
            self.num_features_to_train or num_features_to_train
        )  # channels
        self.input_dimensions = (self.x_timesteps, self.x_height, self.x_width)
        self.input_dimensions_channels = (
            *self.input_dimensions,
            self.num_features_to_train,
        )

        self.y_timesteps = y_timesteps or timesteps
        self.y_height = y_height or height
        self.y_width = y_width or width
        self.num_classes = self.num_classes or num_classes
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
        self.define_model()
        self.define_output_layer()
        self.model = Model(
            inputs=self.input_layer, outputs=self.output_layer, **kwargs
        )
