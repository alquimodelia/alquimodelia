from functools import cached_property
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import ops
from keras.layers import BatchNormalization, Layer, UpSampling2D
from keras.models import Model


class BaseBuilder:
    # Define a self so subclasses can overwrite with inputs having other names (TinEye bands -> num_features_to_train)
    num_features_to_train = None
    num_classes = None

    def _derive_from_input_layer(self, input_shape):
        input_shape = input_shape or ops.shape(self.input_layer)
        # TODO:
        pass

    def define_input_layer(self):
        # This is the base for an input layer. It can be overwrite, but it should set this variable
        # TODO: if there is an input layer, derive the other properties from this
        if self.input_layer:
            self._derive_from_input_layer()
        self.input_layer = self.input_layer or keras.Input(
            self.model_input_shape
        )

    def get_input_layer(self):
        # This is to get the layer to enter the arch, here you can add augmentation or other processing
        input_layer = self.input_layer
        if self.upsampling:
            # TODO: think and set how to get this value
            input_layer = self.UpSampling(
                self.upsampling, data_format=self.data_format
            )(input_layer)
        if self.normalization is not None:
            input_layer = self.normalization()(input_layer)
        return input_layer

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
        activation_end: str = "sigmoid",
        dropout_rate: float = 0.5,
        data_format: str = "channels_last",
        normalization: Layer = None,  # The normalization Layer to apply
        dimensions_to_use=None,
        input_shape: tuple = None,
        input_layer: Layer = None,
        upsampling: int = None,
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

        self.y_timesteps = y_timesteps or timesteps
        self.y_height = y_height or height
        self.y_width = y_width or width
        self.num_classes = self.num_classes or num_classes
        self.output_dimensions = (
            self.y_timesteps,
            self.y_height,
            self.y_width,
        )

        self.activation_end = activation_end
        self.data_format = data_format
        if self.data_format == "channels_first":
            self.channels_dimension = 0
        elif self.data_format == "channels_last":
            self.channels_dimension = -1

        if normalization is None:
            normalization = BatchNormalization
        self.normalization = normalization
        self.upsampling = upsampling
        self.UpSampling = UpSampling2D
        self.dropout_rate = dropout_rate

        self.input_shape = input_shape
        self.input_layer = input_layer

        self.dimensions_to_use = dimensions_to_use  # or ("T", "H", "W", "B")
        self._dimensions_to_use = self.dimensions_to_use

        self.model_setup()
        self.define_input_layer()
        self.define_model()
        self.define_output_layer()
        self.model = Model(
            inputs=self.input_layer, outputs=self.output_layer, **kwargs
        )

    @cached_property
    def model_input_shape(self):
        if self.input_shape:
            input_shape = list(self.input_shape)
            # TODO: make the dimensions if there is an input_shape
            return self.input_shape
        if self._dimensions_to_use:
            self.dimensions_to_use = self._dimensions_to_use
            # This is for a forced dimension use and order.
            input_shape = []
            for dim in self._dimensions_to_use:
                if dim == "T":
                    input_shape.append(self.x_timesteps)
                if dim == "H":
                    input_shape.append(self.x_height)
                if dim == "W":
                    input_shape.append(self.x_width)
                if dim == "B":
                    input_shape.append(self.num_features_to_train)
        else:
            # This defaults to (T, H, W, B). And any (not channels) equal to 1 is droped.
            input_shape = []
            dimension_to_use = []
            for name, size in zip(("T", "H", "W"), self.input_dimensions):
                if size > 1:
                    input_shape.append(size)
                    dimension_to_use.append(name)
            if self.channels_dimension == 0:
                input_shape.insert(0, self.num_features_to_train)
                dimension_to_use.insert(0, "B")
            else:
                input_shape.append(self.num_features_to_train)
                dimension_to_use.append("B")
        self.dimension_to_use = dimension_to_use
        return input_shape

    def opposite_data_format(self):
        if self.data_format == "channels_first":
            return "channels_last"
        elif self.data_format == "channels_last":
            return "channels_first"


class SequenceBuilder(BaseBuilder):
    def __init__(
        self,
        num_sequences:int=1,
        **kwargs,
    ):
        self.num_sequences=num_sequences
        super().__init__(**kwargs)

    def update_kernel(
        self,
        kernel,
        layer_shape,
        data_format="NHWC",  # TODO: mudar pa channels_first?
        kernel_dimension=-2,
        # based on data format # ASsumir que isto esta sempre certo
    ):
        if isinstance(kernel_dimension, int):
            kernel_dimension = [kernel_dimension]
        if isinstance(kernel, int):
            kernel = [kernel]

        max_kernel_tuple_size = len(kernel_dimension)
        if len(kernel) < max_kernel_tuple_size:
            kernel += np.full(max_kernel_tuple_size - len(kernel), max(kernel))

        max_kernel_size = [layer_shape[i] for i in kernel_dimension]

        kernel = tuple([min(m, k) for m, k in zip(max_kernel_size, kernel)])
        # The `kernel_size` argument must be a tuple of 2 integers. Received: (1,)
        if len(kernel) == 1:
            kernel = kernel[0]
        return kernel


    def arch_block(self):
        # Defines the arch block to be used on repetition
        raise NotImplementedError


    def define_model(self):
        # This is the model definition and it must return the output_layer of the architeture. which can be modified further
        # It should use the get_input_layer to fetch the inital layer.
        # it should set the self.last_arch_layer
        input_layer = self.get_input_layer()
        sequence_layer = input_layer
        for i in range(self.num_sequences):
            sequence_layer = self.arch_block(sequence_layer)

        self.last_arch_layer = sequence_layer
        return sequence_layer

    def define_output_layer(self):
        # This should only deal with the last layer, it can be used to define the classification for instance, or multiple methods to close the model
        # it should use the last_arch_layer
        # it should define self.output_layer
        outputDeep = self.last_arch_layer
        self.output_layer = outputDeep

    def model_setup(self):
        # Any needed setup before building and conecting the layers
        pass



class CNN(SequenceBuilder):
    """Base classe for Unet models."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def arch_block(
        self,
        x,
        conv_args=None,
        max_pool_args=None,
        filter_enlarger=4,
        filter_limit=200,
    ):
        """Defines the architecture block for the CNN layer.

        This method defines a block of operations that includes a convolutional layer, a max pooling layer, and a dropout layer.

        Parameters:
        -----------
        x: keras.layer
            Input layer
        conv_args: dict
            Arguments for the convolutional layer. Default is {}.
        max_pool_args: dict
            Arguments for the max pooling layer. Default is {"pool_size": 2}.
        filter_enlarger: int
            Multiplier for the number of filters in the convolutional layer. Default is 4.
        filter_limit: int
            Maximum number of filters in the convolutional layer. Default is 200.

        Returns:
        --------
        x: keras.layer
            Output layer
        """
        if max_pool_args is None:
            max_pool_args = {"pool_size": 2}
        if conv_args is None:
            conv_args = {}
        for k, v in {
            "filters": 16,
            "kernel_size": 3,
            "activation": "relu",
        }.items():
            if k not in conv_args:
                conv_args[k] = v

        x = self.Conv(**conv_args)(x)

        pool = self.update_kernel(max_pool_args["pool_size"], x.shape)
        max_pool_args.update({"pool_size": pool})

        x = self.MaxPooling(**max_pool_args)(x)
        x = self.Dropout(self.dropout_value)(x)

        return x



CNN().model.summary()
print("ss")