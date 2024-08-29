from functools import cached_property

import keras
import numpy as np
from keras import ops
from keras.layers import BatchNormalization, Dense, Layer, UpSampling2D
from keras.models import Model


# TODO: think of initial filters number
class BaseBuilder:
    # Define a self so subclasses can overwrite with inputs having other names (TinEye bands -> num_features_to_train)
    num_features_to_train = None
    num_classes = None

    def _derive_from_input_layer(self, input_shape=None):
        input_shape = input_shape or ops.shape(self.input_layer)
        # TODO:
        pass

    def define_input_layer(self):
        # This is the base for an input layer. It can be overwrite, but it should set this variable
        # TODO: if there is an input layer, derive the other properties from this
        if self.input_layer is not None:
            self._derive_from_input_layer()
        else:
            self.input_layer = keras.Input(self.model_input_shape)

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
        if self.flatten_input:
            input_layer = keras.layers.Flatten()(input_layer)
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
        filters: int = None,
        activation_end: str = "sigmoid",
        activation_middle: str = "relu",
        dropout_rate: float = 0.5,
        data_format: str = "channels_last",
        normalization: Layer = None,  # The normalization Layer to apply
        dimensions_to_use=None,
        dimension_to_predict=None,
        input_shape: tuple = None,
        output_shape: tuple = None,
        input_layer=None,
        upsampling: int = None,
        flatten_input=False,
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
        self.filters = filters
        self.y_timesteps = y_timesteps or timesteps
        self.y_height = y_height or height
        self.y_width = y_width or width
        self.num_classes = self.num_classes or num_classes
        self.output_dimensions = (
            self.y_timesteps,
            self.y_height,
            self.y_width,
        )
        self.activation_middle = activation_middle
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
        self.output_shape = output_shape

        self.input_layer = input_layer
        self.flatten_input = flatten_input

        self.dimensions_to_use = dimensions_to_use  # or ("T", "H", "W", "B")
        self._dimensions_to_use = self.dimensions_to_use
        self.dimension_to_predict = dimension_to_predict
        self._dimension_to_predict = self.dimension_to_predict

        self.model_setup()
        self.define_input_layer()
        self.define_model()
        self.define_output_layer()
        self.model = Model(
            inputs=self.input_layer, outputs=self.output_layer, **kwargs
        )
        self.model.summary()

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
            # Should this be recheked for instances with 1 dim?
            input_shape = []
            dimension_to_use = []
            dict_dimension_to_use = {}
            for name, size in zip(("T", "H", "W"), self.input_dimensions):
                if size > 1:
                    input_shape.append(size)
                    dimension_to_use.append(name)
                    dict_dimension_to_use[name] = size
            if self.channels_dimension == 0:
                input_shape.insert(0, self.num_features_to_train)
                dimension_to_use.insert(0, "B")
                dict_dimension_to_use["B"] = self.num_features_to_train
            else:
                input_shape.append(self.num_features_to_train)
                dimension_to_use.append("B")
                dict_dimension_to_use["B"] = self.num_features_to_train
        self.dimension_to_use = dimension_to_use
        self.dict_dimension_to_use = dict_dimension_to_use
        return tuple(input_shape)

    @cached_property
    def model_output_shape(self):
        if self.output_shape:
            output_shape = list(self.output_shape)
            # TODO: make the dimensions if there is an output_shape
            return self.output_shape
        if self._dimension_to_predict:
            self.dimension_to_predict = self._dimension_to_predict
            # This is for a forced dimension use and order.
            output_shape = []
            for dim in self._dimension_to_predict:
                if dim == "T":
                    output_shape.append(self.y_timesteps)
                if dim == "H":
                    output_shape.append(self.y_height)
                if dim == "W":
                    output_shape.append(self.y_width)
                if dim == "B":
                    output_shape.append(self.num_classes)
        else:
            # This defaults to (T, H, W, B). And any (not channels) equal to 1 is droped.
            # Should this be recheked for instances with 1 dim?
            output_shape = []
            dimension_to_predict = []
            dict_dimension_to_predict = {}
            for name, size in zip(("T", "H", "W"), self.output_dimensions):
                if size > 1:
                    output_shape.append(size)
                    dimension_to_predict.append(name)
                    dict_dimension_to_predict[name] = size
            if self.channels_dimension == 0:
                output_shape.insert(0, self.num_classes)
                dimension_to_predict.insert(0, "B")
                dict_dimension_to_predict["B"] = self.num_classes
            else:
                output_shape.append(self.num_classes)
                dimension_to_predict.append("B")
                dict_dimension_to_predict["B"] = self.num_classes

        self.dimension_to_predict = dimension_to_predict
        self.dict_dimension_to_predict = dict_dimension_to_predict

        return tuple(output_shape)

    def opposite_data_format(self):
        if self.data_format == "channels_first":
            return "channels_last"
        elif self.data_format == "channels_last":
            return "channels_first"


# TODO: review
class SequenceBuilder(BaseBuilder):
    def __init__(
        self,
        num_sequences: int = None,
        flatten_output=True,
        kernel_initializer="he_normal",
        sequence_args=None,
        double_interpretation=True,
        interpretation_filters: int = None,
        **kwargs,
    ):
        self.num_sequences = num_sequences
        self.flatten_output = flatten_output
        self.kernel_initializer = kernel_initializer
        self.sequence_args = sequence_args or {}
        self.interpretation_dense_args = None
        self.double_interpretation = double_interpretation
        self.interpretation_filters = interpretation_filters

        super().__init__(**kwargs)

    # TODO: check how this works in the convs!
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
        sequence_args = self.sequence_args
        if isinstance(self.sequence_args, list):
            assert len(self.sequence_args) == self.num_sequences
        elif isinstance(self.sequence_args, dict):
            sequence_args = [sequence_args] * self.num_sequences
        for i in range(self.num_sequences):
            sequence_layer = self.arch_block(
                sequence_layer, **sequence_args[i]
            )

        self.last_arch_layer = sequence_layer
        return sequence_layer

    def dense_block(
        self,
        x,
        dense_args=None,
        filter_enlarger=4,
        filter_limit=200,
        units_in=None,
        filters_out=None,
        double_dense=True,
    ):
        """Defines the architecture block for the dense layer.

        This method defines a block of operations that includes two dense layers. The number of filters in the dense layers is determined by the `filter_enlarger` and `filter_limit` parameters.

        Parameters:
        -----------
        x: keras.layer
            Input layer
        dense_args: dict or list
            Arguments for the dense layers. If a list, it should contain two dictionaries for the first and second dense layers, respectively.
        filter_enlarger: int
            Multiplier for the number of filters in the dense layers. Default is 4.
        filter_limit: int
            Maximum number of filters in the dense layers. Default is 200.

        Returns:
        --------
        x: keras.layer
            Output layer
        """
        default_dense_args = {"kernel_initializer": self.kernel_initializer}
        out_layer_shape = None
        if dense_args is None:
            dense_args = default_dense_args
        if isinstance(dense_args, list):
            dense_args1 = default_dense_args
            dense_args2 = default_dense_args
            dense_args1.update(dense_args[0])
            units_in = dense_args[0].get("units", None)
            dense_args2.update(dense_args[1])
        else:
            default_dense_args.update(dense_args)
            dense_args1 = default_dense_args
            dense_args2 = default_dense_args

        filters_out = dense_args2.pop("units", filters_out)
        # filters_out = filters_out or self.num_classes
        if filters_out is None:
            if self.flatten_output or len(x.shape) == 2:
                filters_out = np.prod(self.model_output_shape)
            else:
                filters_out = self.model_output_shape[-1]
        units_in = units_in or self.interpretation_filters
        if units_in is None:
            units_in = dense_args1.pop(
                "units",
                min(filters_out * filter_enlarger, filter_limit),
            )
        if double_dense:
            if units_in > x.shape[-1]:
                out_layer_shape = list(x.shape[1:])
                out_layer_shape[-1] = filters_out
                x = keras.layers.Flatten()(x)
                filters_out = np.prod(self.model_output_shape)
            x = Dense(units_in, **dense_args1)(x)
        x = Dense(filters_out, **dense_args2)(x)
        if out_layer_shape is not None:
            x = keras.layers.Reshape(out_layer_shape)(x)

        return x

    def interpretation_layer(
        self,
        output_layer,
        dense_args=None,
        output_layer_args=None,
        double_interpretation=None,
    ):
        """Defines the interpretation layers for the model.

        This method defines a block of operations that includes a dense layer and an output layer. The arguments for the dense layer are determined by the `dense_args` parameter.

        Parameters:
        -----------
        output_layer: keras.layer
            Input layer for the interpretation layers
        dense_args: dict
            Arguments for the dense layer. Default is None, which means to use the default arguments.
        output_layer_args: dict
            Arguments for the output layer. Default is {}.

        Returns:
        --------
        output_layer: keras.layer
            Output layer
        """
        if output_layer_args is None:
            output_layer_args = {}
        dense_args = self.interpretation_dense_args
        double_interpretation = (
            double_interpretation or self.double_interpretation
        )

        if dense_args is None:
            dense_args = {}
            if self.activation_end != self.activation_middle:
                dense_args = [
                    {"activation": self.activation_middle},
                    {"activation": self.activation_end},
                ]
            else:
                if isinstance(dense_args, list):
                    dense_args[0].update(
                        {"activation": self.activation_middle}
                    )
                    dense_args[1].update({"activation": self.activation_end})
                else:
                    dense_args.update({"activation": self.activation_end})

        output_layer = self.dense_block(
            output_layer,
            dense_args=dense_args,
            double_dense=double_interpretation,
        )
        return output_layer

    def define_output_layer(self):
        # This should only deal with the last layer, it can be used to define the classification for instance, or multiple methods to close the model
        # it should use the last_arch_layer
        # it should define self.output_layer
        outputDeep = self.last_arch_layer
        if self.flatten_output:
            outputDeep = keras.layers.Flatten()(outputDeep)
        outputDeep = self.interpretation_layer(outputDeep)
        if outputDeep.shape[1:] != self.model_output_shape:
            outputDeep = keras.layers.Reshape(self.model_output_shape)(
                outputDeep
            )
        self.output_layer = outputDeep
