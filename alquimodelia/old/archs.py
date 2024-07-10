import math
import os

os.environ["KERAS_BACKEND"] = "torch"


import numpy as np
from keras import Model, ops
from keras.activations import gelu
from keras.layers import (
    LSTM,
    Add,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LayerNormalization,
    MaxPooling1D,
    MaxPooling2D,
    MaxPooling3D,
    MultiHeadAttention,
    Normalization,
    RepeatVector,
    Reshape,
    TimeDistributed,
)

from alquimodelia.unet import AttResUNet1D
from alquimodelia.utils import (
    adjust_to_multiple,
    stays_if_not_bigger,
    stays_if_not_smaller,
)


class ForeArch:
    """Base class for different architectures.

    Attributes:
    -----------
    X_timeseries: int
        Number of time series in the input
    Y_timeseries: int
        Number of time series in the output
    n_features_train: int
        Number of features in the training set
    n_features_predict: int
        Number of features in the targer set
    dropout: float
        Dropout rate
    dense_out: int
        Number of output neurons in the dense layer
    input_shape: tuple
        Shape of the input layer
    output_shape: tuple
        Shape of the output layer
    """

    def __init__(
        self,
        X_timeseries,
        Y_timeseries,
        n_features_train,
        n_features_predict,
        dropout=0.35,
        activation_end="relu",
        activation_middle="relu",
        kernel_initializer="he_normal",
    ):
        self.X_timeseries = X_timeseries
        self.Y_timeseries = Y_timeseries
        self.n_features_train = n_features_train
        self.n_features_predict = n_features_predict
        self.dense_out = self.Y_timeseries * self.n_features_predict
        self.set_input_shape()
        self.set_output_shape()
        self.dropout_value = dropout
        self.activation_end = activation_end
        self.activation_middle = activation_middle
        self.kernel_initializer = kernel_initializer

    def set_input_shape(self):
        """Sets the input shape."""
        self.input_shape = (self.X_timeseries, self.n_features_train)

    def set_output_shape(self):
        """Sets the output shape."""
        self.output_shape = (self.Y_timeseries, self.n_features_predict)

    def interpretation_layers(self):
        pass

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

        kernel = tuple(
            [
                stays_if_not_bigger(m, k)
                for m, k in zip(max_kernel_size, kernel)
            ]
        )
        # The `kernel_size` argument must be a tuple of 2 integers. Received: (1,)
        if len(kernel) == 1:
            kernel = kernel[0]
        return kernel

    def get_time2vec(
        self,
        input_layer,
        time2vec_kernel_size=None,
    ):
        from alquitable.layers import Time2Vec

        if not isinstance(time2vec_kernel_size, list):
            time2vec_kernel_size = [time2vec_kernel_size]

        timed_layers = []
        for kernel_size in time2vec_kernel_size:
            time2vec = Time2Vec(kernel_size=kernel_size)
            # TODO: study this better
            timed_layer = TimeDistributed(time2vec)(input_layer)
            if self.dropout_value:
                timed_layer = Dropout(self.dropout_value)(timed_layer)
            timed_layers.append(timed_layer)

        input_layer = ops.concatenate([input_layer, *timed_layers], -1)

        return input_layer

    def get_input_layer(
        self,
        normalization=True,
        bacth_norm=True,
        flatten_input=False,
        time2vec_kernel_size=None,
        timedist=False,
    ):
        """Returns the input layer with optional normalization and flattening.

        Parameters:
        -----------
        normalization: bool
            If True, applies normalization to the input layer.
        batch_norm: bool
            If True, applies batch normalization to the input layer.
        flatten_input: bool
            If True, applies flattening to the input layer.

        Returns:
        --------
        input_layer: keras.layer
            Input layer.
        """
        input_layer = Input(self.input_shape)
        self.input_layer = input_layer
        if normalization:
            if bacth_norm:
                input_layer = BatchNormalization()(input_layer)
            else:
                input_layer = Normalization()(input_layer)

        if time2vec_kernel_size:
            input_layer = self.get_time2vec(input_layer, time2vec_kernel_size)

        if flatten_input:
            input_layer = Flatten()(input_layer)

        return input_layer

    def get_output_layer(self, out_layer, reshape_shape=None):
        """Returns the output layer with optional reshaping.

        Parameters:
        -----------
        out_layer: tensorflow.python.keras.engine.keras_tensor.KerasTensor
            Output layer
        reshape_shape: tuple
            Shape to reshape the output layer

        Returns:
        --------
        out_layer: tensorflow.python.keras.engine.keras_tensor.KerasTensor
            Output layer
        """
        if reshape_shape is None:
            reshape_shape = self.output_shape

        if reshape_shape is not None:
            out_layer = Reshape(reshape_shape)(out_layer)
        return out_layer

    def arch_block(self, input_layer):
        """Architecture block to be implemented in subclasses."""
        pass

    def stacked_repetition(
        self, input_layer, block_repetition=1, block_args=None
    ):
        if block_args is None:
            block_args = {}
        args_use = block_args
        for i in range(block_repetition):
            if isinstance(block_args, list):
                args_use = block_args[i]
            input_layer = self.arch_block(input_layer, **args_use)

        return input_layer

    def paralel_repetition(
        self, input_layer, block_repetition=1, block=None, block_args=None
    ):
        if not block:
            block = self.arch_block

        output_layer = list()
        block_in = input_layer
        block_args_in = block_args

        for i in range(block_repetition):
            if isinstance(input_layer, list):
                block_in = input_layer[i]
            if isinstance(block_args, list):
                block_args_in = block_args[i]
            output_layer.append(block(block_in, **block_args_in))

        return output_layer

    def multihead(
        self,
        input_layers,
        block_repetition=1,
        dim_split=None,
        reshape_into_original_dim_size=True,
        block=None,
    ):
        if isinstance(input_layers, list):
            block_repetition = len(input_layers)
            layer_shape = input_layers[0].shape
        else:
            layer_shape = input_layers.shape
        # if dim split not specified it will try to get the dimension
        # where to split
        if not dim_split:
            for i, dim in enumerate(layer_shape):
                if dim == block_repetition:
                    dim_split = i

        inputs = []
        for i in range(block_repetition):
            # If it comes as a list, get the index
            if isinstance(input_layers, list):
                x = ops.gather(input_layers, indices=i, axis=dim_split)
                # x = K.gather(input_layers, indices=i#, axis=dim_split
                # )
                if reshape_into_original_dim_size:
                    in_shape = (*x.shape[1:], 1)
                    x = Reshape(in_shape)(x)

            # Otherwise repeats the input_layer
            else:
                x = input_layers
            inputs.append(x)

        outputs = self.paralel_repetition(
            inputs, block_repetition=block_repetition, block=block
        )

        return outputs

    def architecture(self):
        pass

    def mlp(self, x, hidden_units, dropout_rate=None, activation=gelu):
        dropout_rate = dropout_rate or self.dropout_value
        for units in hidden_units:
            x = Dense(units, activation=activation)(x)
            x = Dropout(dropout_rate)(x)
        return x

    def mlp_strategy(
        self,
        x,
        mlp_head_units=None,
        strategy="dimwise",
        out_shape=None,
        num_hidden_units=2,
        **mlp_args
    ):
        out_shape = out_shape or self.output_shape
        original_x_shape = x.shape
        # 2) drop the dimension to the final one (reshapes, leave the n_features_to_predict out of this)
        if strategy == "dimwise":
            num_dims = len(original_x_shape)
            max_dim = num_dims - 1
            list_dims = [f for f in range(1, max_dim)]
            transpose_positions = [0, max_dim, *list_dims]
            for dim in list_dims:
                x = ops.transpose(x, transpose_positions)
                x = Dense(out_shape[(dim - 1)])(x)
            x = ops.transpose(x, transpose_positions)
        else:
            # 1) given mlp_units
            if not mlp_head_units:
                output_size = np.prod(self.output_shape)
                # 4) [output_size*((f+1)**2) for f in range]~~~
                x = Flatten()(x)
                if strategy == "square":
                    mlp_head_units = [
                        output_size * ((f + 2) ** 2)
                        for f in range(num_hidden_units)
                    ]
                    mlp_head_units.reverse()
                elif strategy == "logarithmic":
                    # 3) log2 steps
                    representation_size_log2 = int(math.log2(x.shape[-1]))
                    final_size_log2 = max(int(math.log2(output_size)), 1)
                    mlp_head_units_steps = int(
                        (representation_size_log2 - final_size_log2)
                        / (num_hidden_units + 1)
                    )
                    B_value = 2 ** (
                        representation_size_log2 - (mlp_head_units_steps * 2)
                    )
                    # This is the one I like the most, it balance the power of two betweent the flatten values
                    mlp_head_units = [
                        2 ** (representation_size_log2 - mlp_head_units_steps),
                        adjust_to_multiple(B_value, self.Y_timeseries),
                    ]  # 42s
                x = self.mlp(x, mlp_head_units, **mlp_args)
                last_output_shape = (
                    self.Y_timeseries,
                    int(x.shape[-1] / self.Y_timeseries),
                )
                x = Reshape(last_output_shape)(x)

        return x


class DenseArch(ForeArch):
    """This architecture uses a dense layer to solve the problem.

    This class inherits from the `ForeArch` class and overrides the `arch_block` and `architecture` methods to implement a dense layer architecture.
    """

    def arch_block(
        self, x, dense_args=None, filter_enlarger=4, filter_limit=200
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
        units_in = None
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

        filters_out = dense_args2.pop("units", None)
        if filters_out is None:
            filters_out = self.dense_out
        if units_in is None:
            units_in = dense_args1.pop(
                "units",
                stays_if_not_smaller(
                    filters_out * filter_enlarger, filter_limit
                ),
            )

        x = Dense(
            units_in,
            **dense_args1,
        )(x)
        x = Dense(
            filters_out,
            **dense_args2,
        )(x)

        return x

    def architecture(
        self,
        block_repetition=1,
        dense_args=None,
        get_input_layer_args=None,
        **kwargs
    ):
        """Defines the architecture of the model.

        This method defines the architecture of the model, which includes an input layer, an architecture block, and an output layer.

        Returns:
        --------
        model: keras.Model
            Keras model with the defined architecture
        """
        get_input_layer_args = get_input_layer_args or {}

        input_layer = self.get_input_layer(
            flatten_input=True, **get_input_layer_args
        )
        output_layer = self.arch_block(input_layer)
        block_args = {"dense_args": dense_args}
        if block_repetition == 1:
            output_layer = self.arch_block(input_layer, dense_args=dense_args)
        elif block_repetition > 1:
            if not dense_args:
                dense_args = {}
                block_args["dense_args"] = dense_args
            if "units" not in dense_args:
                num_filters = self.dense_out
            else:
                num_filters = dense_args["units"]
            if not isinstance(num_filters, list):
                num_filters = [
                    {"dense_args": {"units": int(num_filters * (4**f))}}
                    for f in np.arange(block_repetition)
                ]
                num_filters.reverse()
            if isinstance(num_filters, list):
                un2_last = None
                for i in range(len(num_filters)):
                    unit2 = num_filters[i]["dense_args"]["units"]
                    unit1 = stays_if_not_smaller(unit2 * 4, 200)
                    if un2_last:
                        if unit1 > un2_last:
                            rat = math.sqrt(un2_last / unit2)
                            unit1 = int(unit2 * rat)
                            num_filters[i]["dense_args"] = [
                                {"units": unit1},
                                {"units": unit2},
                            ]
                    un2_last = unit2

            block_args = num_filters
            output_layer = self.stacked_repetition(
                input_layer, block_repetition, block_args=block_args
            )

        output_layer = self.get_output_layer(output_layer)

        return Model(inputs=self.input_layer, outputs=output_layer, **kwargs)


class LSTMArch(ForeArch):
    """This class is LSTM layer architecture."""

    def arch_block(
        self, x, lstm_args=None, filter_enlarger=4, filter_limit=200
    ):
        """Defines the architecture block for the LSTM layer.

        This method defines a block of operations that includes an LSTM layer. The arguments for the LSTM layer are determined by the `lstm_args` parameter.

        Parameters:
        -----------
        x: keras.layer
            Input layer
        lstm_args: dict
            Arguments for the LSTM layer. Default is {"units": 50, "activation": "relu"}.
        filter_enlarger: int
            Multiplier for the number of filters in the dense layers. Default is 4.
        filter_limit: int
            Maximum number of filters in the dense layers. Default is 200.

        Returns:
        --------
        x: keras.layer
            Output layer
        """
        # Default LSTM arguments
        if lstm_args is None:
            lstm_args = {"units": 50, "activation": "relu"}
        default_lstm_args = {"units": 50, "activation": "relu"}

        # If lstm_args is provided, update the default arguments
        if lstm_args is not None:
            default_lstm_args.update(lstm_args)

        # Apply LSTM layer
        x = LSTM(**default_lstm_args)(x)
        x = Dropout(self.dropout_value)(x)
        return x

    def interpretation_layers(
        self, output_layer, dense_args=None, output_layer_args=None
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
        if dense_args is None:
            dense_args = {}
            if self.activation_end != self.activation_middle:
                dense_args = [
                    {"activation": self.activation_middle},
                    {"activation": self.activation_end},
                ]
            else:
                dense_args = {"activation": self.activation_end}

        output_layer = DenseArch.arch_block(
            self, output_layer, dense_args=dense_args
        )
        output_layer = self.get_output_layer(output_layer, **output_layer_args)
        return output_layer

    def architecture(
        self,
        block_repetition=1,
        dense_args=None,
        block_args=None,
        get_input_layer_args=None,
        **kwargs
    ):
        """Defines the architecture of the model.

        This method defines the architecture of the model, which includes an input layer, an architecture block, interpretation layers, and an output layer.

        Parameters:
        -----------
        block_repetition: int
            Number of times to repeat the architecture block. Default is 1.
        dense_args: dict
            Arguments for the dense layer in the interpretation layers. Default is {"activation": "softplus"}.
        block_args: dict or list
            Arguments for the architecture block. If a list, it should contain a dictionary for each repetition of the block.

        Returns:
        --------
        model: keras.Model
            Keras model with the defined architecture
        """
        get_input_layer_args = get_input_layer_args or {}

        if dense_args is None:
            dense_args = {"activation": "softplus"}
        input_layer = self.get_input_layer(
            flatten_input=False, **get_input_layer_args
        )
        if block_repetition == 1:
            output_layer = self.arch_block(input_layer)
        elif block_repetition > 1:
            if block_args is None:
                block_args = [{"lstm_args": {"return_sequences": True}}, {}]
            output_layer = self.stacked_repetition(
                input_layer, block_repetition, block_args=block_args
            )
        output_layer = self.interpretation_layers(output_layer, dense_args)
        return Model(inputs=self.input_layer, outputs=output_layer, **kwargs)


class CNNArch(ForeArch):
    """This architecture uses a convolutional neural network (CNN) layer to solve the problem.

    This class inherits from the `ForeArch` class and overrides the `arch_block`, `interpretation_layers`, and `architecture` methods to implement a CNN layer architecture.
    """

    def __init__(self, conv_dimension="1D", **kwargs):
        """Initializes the CNNArch class.

        Parameters:
        -----------
        conv_dimension: str
            Dimension of the convolutional layer. It can be "1D", "2D", or "3D". Default is "1D".
        **kwargs: dict
            Additional keyword arguments for the `ForeArch` class.
        """
        self.set_dimension_layer(conv_dimension)
        super().__init__(**kwargs)

    def set_dimension_layer(self, conv_dimension):
        """Sets the dimension of the convolutional layer.

        This method sets the convolutional layer, max pooling layer, and dropout layer based on the dimension of the convolutional layer.

        Parameters:
        -----------
        conv_dimension: str
            Dimension of the convolutional layer. It can be "1D", "2D", or "3D".
        """
        if conv_dimension == "1D":
            self.MaxPooling = MaxPooling1D
            self.Conv = Conv1D
            self.Dropout = Dropout
        elif conv_dimension == "2D":
            self.MaxPooling = MaxPooling2D
            self.Conv = Conv2D
            self.Dropout = Dropout
        elif conv_dimension == "3D":
            self.MaxPooling = MaxPooling3D
            self.Conv = Conv3D
            self.Dropout = Dropout

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

    def interpretation_layers(
        self, output_layer, dense_args=None, output_layer_args=None
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
        if dense_args is None:
            dense_args = {}
            if self.activation_end != self.activation_middle:
                dense_args = [
                    {"activation": self.activation_middle},
                    {"activation": self.activation_end},
                ]
            else:
                dense_args = {"activation": self.activation_end}

        output_layer = DenseArch.arch_block(
            self, output_layer, dense_args=dense_args
        )
        output_layer = self.get_output_layer(output_layer, **output_layer_args)
        return output_layer

    def architecture(
        self,
        block_repetition=1,
        multitail=False,
        conv_args=None,
        get_input_layer_args=None,
        **kwargs
    ):
        """Defines the architecture of the model.

        This method defines the architecture of the model, which includes an input layer, an architecture block, interpretation layers, and an output layer. The architecture block can be repeated multiple times as specified by the `block_repetition` parameter. The `multitail` parameter determines whether to use a parallel repetition of the interpretation layers.

        Parameters:
        -----------
        block_repetition: int
            Number of times to repeat the architecture block. Default is 1.
        multitail: bool or int or list
            If True or an integer, uses a parallel repetition of the interpretation layers. The number of repetitions is determined by the value of `multitail`. If a list, it should contain the arguments for each repetition of the interpretation layers. Default is False, which means to not use parallel repetition.

        Returns:
        --------
        model: keras.Model
            Keras model with the defined architecture
        """
        get_input_layer_args = get_input_layer_args or {}

        input_layer = self.get_input_layer(
            flatten_input=False, **get_input_layer_args
        )
        block_args = {"conv_args": conv_args}
        if block_repetition == 1:
            output_layer = self.arch_block(input_layer, conv_args=conv_args)
        elif block_repetition > 1:
            if conv_args:
                if "filters" not in conv_args:
                    num_filters = [
                        {"conv_args": {"filters": 2**f}}
                        for f in np.arange(block_repetition)
                    ]
                    num_filters.reverse()
                else:
                    num_filters = conv_args["filters"]
                    if not isinstance(num_filters, list):
                        num_filters = [
                            {
                                "conv_args": {
                                    "filters": int(num_filters * (2**f))
                                }
                            }
                            for f in np.arange(block_repetition)
                        ]
                        num_filters.reverse()
                block_args = num_filters
            else:
                conv_args = {}
                block_args["conv_args"] = conv_args
            output_layer = self.stacked_repetition(
                input_layer, block_repetition, block_args=block_args
            )
        output_layer = Flatten()(output_layer)

        if multitail is not False:
            if isinstance(multitail, list):
                multitail_repetition = len(multitail)
            elif isinstance(multitail, int):
                multitail_repetition = multitail
            else:
                multitail_repetition = 1

            output_layer = self.paralel_repetition(
                output_layer,
                multitail_repetition,
                self.interpretation_layers,
                block_args=multitail,
            )
        else:
            output_layer = self.interpretation_layers(output_layer)

        return Model(inputs=self.input_layer, outputs=output_layer, **kwargs)


class UNETArch(ForeArch):
    """This architecture just follow the idea of a dense layers to solve the problem."""

    def __init__(self, conv_dimension="1D", **kwargs):
        self.set_dimension_layer(conv_dimension)
        super().__init__(**kwargs)

    def set_dimension_layer(self, conv_dimension):
        if conv_dimension == "1D":
            self.MaxPooling = MaxPooling1D
            self.Conv = Conv1D
            self.Dropout = Dropout
        elif conv_dimension == "2D":
            self.MaxPooling = MaxPooling2D
            self.Conv = Conv2D
            self.Dropout = Dropout
        elif conv_dimension == "3D":
            self.MaxPooling = MaxPooling3D
            self.Conv = Conv3D
            self.Dropout = Dropout

    def architecture(
        self, conv_args=None, get_input_layer_args=None, **kwargs
    ):
        get_input_layer_args = get_input_layer_args or {}

        if conv_args is None:
            conv_args = {"filters": 16}
        model_UNET = AttResUNet1D(
            width=self.X_timeseries,
            num_bands=self.n_features_train,
            data_format="channels_last",
            n_filters=conv_args["filters"],
            num_classes=self.n_features_predict,
            activation_end=self.activation_end,
            activation_middle=self.activation_middle,
        )

        x = Model(
            inputs=model_UNET.input_layer,
            outputs=model_UNET.output_layer,
            **kwargs,
        )

        return x


class EncoderDecoder(LSTMArch):
    """This architecture uses an Encoder-Decoder LSTM network to solve the problem.

    This class inherits from the `LSTMArch` class and overrides the `arch_block` and `interpretation_layers` methods to implement an Encoder-Decoder LSTM network architecture.
    """

    def arch_block(
        self,
        x,
        lstm_args=None,
        filter_enlarger=4,
        filter_limit=200,
    ):
        """Defines the architecture block for the Encoder-Decoder LSTM network.

        This method defines a block of operations that includes an LSTM layer as the encoder, a RepeatVector layer, another LSTM layer as the decoder, and a Dropout layer.

        Parameters:
        -----------
        x: keras.layer
            Input layer
        lstm_args: dict
            Arguments for the LSTM layers. Default is {"units": 50, "activation": "relu"}.
        filter_enlarger: int
            Multiplier for the number of filters in the LSTM layers. Default is 4.
        filter_limit: int
            Maximum number of filters in the LSTM layers. Default is 200.

        Returns:
        --------
        x: keras.layer
            Output layer
        """
        # Default LSTM arguments
        if lstm_args is None:
            lstm_args = {"units": 50, "activation": "relu"}
        default_lstm_args = {"units": 50, "activation": "relu"}

        # If lstm_args is provided, update the default arguments
        if lstm_args is not None:
            default_lstm_args.update(lstm_args)

        # Apply LSTM layer (Encoder)
        x = LSTM(**default_lstm_args)(x)

        # RepeatVector layer
        x = RepeatVector(self.Y_timeseries)(x)

        # Apply LSTM layer (Decoder)
        x = LSTM(**default_lstm_args, return_sequences=True)(x)

        # Apply Dropout layer
        x = Dropout(self.dropout_value)(x)

        return x

    def interpretation_layers(self, x, dense_args):
        """Defines the interpretation layers for the model.

        This method defines a block of operations that includes a TimeDistributed layer with a Dense layer, a Dropout layer, and an output layer. The arguments for the Dense layer are determined by the `dense_args` parameter.

        Parameters:
        -----------
        x: keras.layer
            Input layer for the interpretation layers
        dense_args: dict
            Arguments for the Dense layer.

        Returns:
        --------
        x: keras.layer
            Output layer
        """
        time_dim_size = x.shape[1]
        dense_filters = int(self.dense_out / time_dim_size)
        # Apply TimeDistributed layer with Dense layer
        x = TimeDistributed(Dense(dense_filters, **dense_args))(x)

        # Apply Dropout layer
        x = Dropout(self.dropout_value)(x)

        # Apply output layer
        x = self.get_output_layer(x)

        return x


class Transformer(ForeArch):
    def arch_block(
        self,
        input_layer,
        projection_dim=None,
        transformer_units=None,
        num_heads=8,
    ):
        projection_dim = projection_dim or self.n_features_train
        transformer_units = transformer_units or [
            projection_dim * 2,
            projection_dim,
        ]
        if transformer_units[-1] != projection_dim:
            # TODO: warning that we are changing this, because the transformer block needs to finisehd with the same size
            transformer_units[-1] = projection_dim

        # Layer normalization 1.
        x1 = LayerNormalization()(input_layer)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, input_layer])
        # Layer normalization 2.
        x3 = LayerNormalization()(x2)
        # MLP.
        x3 = self.mlp(
            x3,
            hidden_units=transformer_units,
            dropout_rate=0.1,
            activation=self.activation_middle,
        )
        # Skip connection 2.
        input_layer = Add()([x3, x2])
        return input_layer

    def get_decoder(self, logits):
        return

    def interpretation_layers(
        self, x, interpretation_strategy="mlp", **interpretation_strategy_args
    ):
        if interpretation_strategy == "mlp":
            x = self.mlp_strategy(x, **interpretation_strategy_args)
        elif interpretation_strategy == "cnn":
            # TODO: todo
            x = CNNArch.arch_block(self, x, **interpretation_strategy_args)
        x = Dense(self.n_features_predict)(x)
        return x

    def architecture(
        self,
        block_repetition=1,
        get_input_layer_args=None,
        arch_block_args=None,
        interpretation_layers_args=None,
        **kwargs
    ):
        get_input_layer_args = get_input_layer_args or {}
        arch_block_args = arch_block_args or {}
        interpretation_layers_args = interpretation_layers_args or {}

        input_layer = self.get_input_layer(**get_input_layer_args)

        for _ in range(block_repetition):
            # TODO: change this in a way that you can define the archs per iteration
            input_layer = self.arch_block(input_layer, **arch_block_args)

        # logits = create_vit_classifier(
        #     input_layer,
        #     augmentation=False,
        #     projection_dim=self.n_features_train,
        #     activation_end=self.activation_end,
        #     activation_middle=self.activation_middle,
        #     transformer_layers=block_repetition,
        #     n_features_predict=self.n_features_predict,
        #     Y_timeseries=self.Y_timeseries,
        # )
        output = self.interpretation_layers(
            input_layer, **interpretation_layers_args
        )

        model = Model(inputs=self.input_layer, outputs=output, **kwargs)
        return model
