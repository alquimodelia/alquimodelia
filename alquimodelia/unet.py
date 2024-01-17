# This module is for internal use only, it should be in another
#   package, specialized in images

from functools import cached_property
from typing import Any, Dict

import keras as keras
from keras import ops
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Conv1DTranspose,
    Cropping2D,
    Dropout,
    MaxPooling1D,
    Multiply,
    Reshape,
    UpSampling1D,
    concatenate,
)
from keras.src.legacy.backend import int_shape


def count_number_divisions(size: int, count: int, by: int = 2, limit: int = 2):
    """
    Count the number of possible steps.

    Parameters
    ----------
    size : int
        Image size (considering it is a square).
    count : int
        Input must be 0.
    by : int, optional
        The factor by which the size is divided. Default is 2.
    limit : int, optional
        Size of last filter (smaller). Default is 2.

    Returns
    -------
    int
        The number of possible steps.
    """
    if size >= limit:
        if size % 2 == 0:
            count = count_number_divisions(
                size / by, count + 1, by=by, limit=limit
            )
    else:
        count = count - 1
    return count


class ModelMagia(keras.Model):
    def model_input_shape(self):
        raise NotImplementedError

    def define_input_layer(self):
        self.input_layer = keras.Input(self.model_input_shape)

    def output_layer(self):
        raise NotImplementedError

    def get_last_layer_activation(self):
        raise NotImplementedError

    def __init__(
        self,
        timesteps: int = 0,
        width: int = 0,
        height: int = 0,
        num_bands: int = 0,
        num_classes: int = 0,
        activation_final: str = "sigmoid",
        data_format: str = "channels_last",
    ):
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.width = width
        self.height = height
        self.num_bands = num_bands
        self.activation_final = activation_final
        self.data_format = data_format
        self.define_input_layer()

        super().__init__(
            inputs=self.input_layer,
            outputs=self.output_layer(),
        )


def repeat_elem(tensor, rep):
    return ops.tile(tensor, [1, 1, rep])


class UNet(ModelMagia):
    """Base classe for Unet models."""

    def __init__(
        self,
        n_filters: int = 16,
        number_of_conv_layers: int = 0,
        kernel_size: int = 3,
        batchnorm: bool = True,
        padding_style: str = "same",
        padding: int = 0,
        activation_middle: str = "relu",
        activation_end: str = "softmax",
        kernel_initializer: str = "he_normal",
        dropout: float = 0.5,
        **kwargs,
    ):
        self._number_of_conv_layers = number_of_conv_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batchnorm = batchnorm
        self.padding = padding
        self.padding_style = padding_style
        self.dropout = dropout

        self.activation_middle = activation_middle
        self.activation_end = activation_end
        self.kernel_initializer = kernel_initializer
        super().__init__(**kwargs)

    @cached_property
    def number_of_conv_layers(self):
        if self._number_of_conv_layers == 0:
            number_of_layers = []
            if self.data_format == "channels_first":
                study_shape = self.model_input_shape[1:]
            elif self.data_format == "channels_last":
                study_shape = self.model_input_shape[:-1]
            for size in study_shape:
                number_of_layers.append(count_number_divisions(size, 0))

            self._number_of_conv_layers = min(number_of_layers)
            self._number_of_conv_layers = max(self._number_of_conv_layers, 1)

        return self._number_of_conv_layers

    def opposite_data_format(self):
        if self.data_format == "channels_first":
            return "channels_last"
        elif self.data_format == "channels_last":
            return "channels_first"

    def convolution_block(
        self,
        input_tensor,
        n_filters: int,
        kernel_size: int = 3,
        batchnorm: bool = True,
        data_format: str = "channels_first",
        padding: str = "same",
        activation: str = "relu",
        kernel_initializer: str = "he_normal",
    ):
        # first layer
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        # Second layer.
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(x)
        if batchnorm:
            x = BatchNormalization()(x)
        return x

    def contracting_block(
        self,
        input_img,
        n_filters: int = 16,
        batchnorm: bool = True,
        dropout: float = 0.25,
        kernel_size: int = 3,
        strides: int = 2,
        data_format: str = "channels_last",
        padding: str = "same",
        activation: str = "relu",
    ):
        c1 = self.convolution_block(
            input_img,
            n_filters=n_filters,
            kernel_size=kernel_size,
            batchnorm=batchnorm,
            data_format=data_format,
            activation=activation,
            padding=padding,
        )
        p1 = self.MaxPooling(strides, padding=padding)(c1)
        p1 = self.SpatialDropout(dropout)(p1)
        return p1, c1

    def expansive_block(
        self,
        ci,
        cii,
        n_filters: int = 16,
        batchnorm: bool = True,
        dropout: float = 0.5,
        kernel_size: int = 3,
        strides: int = 2,
        data_format: str = "channels_first",
        activation: str = "relu",
        padding_style: str = "same",
    ):
        u = self.ConvTranspose(
            n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding_style,
            data_format=data_format,
        )(ci)
        u = concatenate([u, cii])
        u = self.SpatialDropout(dropout)(u)
        c = self.convolution_block(
            u,
            n_filters=n_filters,
            kernel_size=kernel_size,
            batchnorm=batchnorm,
            data_format=data_format,
            activation=activation,
            padding=padding_style,
        )
        return c

    def contracting_loop(
        self, input_img, contracting_arguments: Dict[str, Any]
    ):
        list_p = [input_img]
        list_c = []
        n_filters = contracting_arguments["n_filters"]
        for i in range(self.number_of_conv_layers + 1):
            old_p = list_p[i]
            filter_expansion = 2**i
            contracting_arguments["n_filters"] = n_filters * filter_expansion
            p, c = self.contracting_block(old_p, **contracting_arguments)
            list_p.append(p)
            list_c.append(c)
        return list_c

    def expanding_loop(
        self, contracted_layers, expansion_arguments: Dict[str, Any]
    ):
        list_c = [contracted_layers[-1]]
        iterator_expanded_blocks = range(self.number_of_conv_layers)
        iterator_contracted_blocks = reversed(iterator_expanded_blocks)
        n_filters = expansion_arguments["n_filters"]
        for i, c in zip(iterator_expanded_blocks, iterator_contracted_blocks):
            filter_expansion = 2 ** (c)
            expansion_arguments["n_filters"] = n_filters * filter_expansion
            c4 = self.expansive_block(
                list_c[i], contracted_layers[c], **expansion_arguments
            )
            list_c.append(c4)
        return c4

    def deep_neural_network(
        self,
        n_filters: int = 16,
        dropout: float = 0.2,
        batchnorm: bool = True,
        data_format: str = "channels_last",
        activation_middle: str = "relu",
        kernel_size: int = 3,
        padding: str = "same",
    ):
        """Build deep neural network."""
        input_img = self.input_layer
        # self.define_number_convolution_layers()

        contracting_arguments = {
            "n_filters": n_filters,
            "batchnorm": batchnorm,
            "dropout": dropout,
            "kernel_size": kernel_size,
            "padding": padding,
            "data_format": data_format,
            "activation": activation_middle,
        }
        expansion_arguments = {
            "n_filters": n_filters,
            "batchnorm": batchnorm,
            "dropout": dropout,
            "data_format": data_format,
            "activation": activation_middle,
            "kernel_size": kernel_size,
        }

        contracted_layers = self.contracting_loop(
            input_img, contracting_arguments
        )
        unet_output = self.expanding_loop(
            contracted_layers, expansion_arguments
        )

        return unet_output

    def gating_signal(self, input_tensor, out_size, batch_norm=True):
        """
        Resize the down layer feature map into the same dimension as the up
        layer feature map using 1x1 conv.

        Parameters
        ----------
        input_tensor: keras.layer
            The input layer to be resized.
        out_size: int
            The size of the output layer.
        batch_norm: bool, optional
            If True, applies batch normalization to the input layer.
            Default is True.

        Returns
        -------
        keras.layer
            The gating feature map with the same dimension as the up layer
            feature map.
        """
        # first layer
        x = self.Conv(
            filters=out_size,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            padding="same",
            data_format=self.data_format,
            activation="relu",
        )(input_tensor)

        x = BatchNormalization()(x)
        return x

    def attention_block(self, x, gating, inter_shape):
        shape_x = int_shape(x)
        shape_g = int_shape(gating)

        # Getting the x signal to the same shape as the gating signal
        theta_x = self.Conv(inter_shape, 2, strides=2, padding="same")(x)  # 16
        shape_theta_x = int_shape(theta_x)

        # Getting the gating signal to the same number of filters
        #   as the inter_shape
        phi_g = self.Conv(inter_shape, 1, padding="same")(gating)
        upsample_g = self.ConvTranspose(
            inter_shape,
            3,
            strides=(shape_theta_x[1] // shape_g[1]),
            padding="same",
        )(
            phi_g
        )  # 16

        concat_xg = Add()([upsample_g, theta_x])
        act_xg = Activation("relu")(concat_xg)
        psi = self.Conv(1, 1, padding="same")(act_xg)
        sigmoid_xg = Activation("sigmoid")(psi)
        shape_sigmoid = int_shape(sigmoid_xg)
        sss = (shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2])
        upsample_psi = self.UpSampling(size=sss[0])(sigmoid_xg)  # 32
        upsample_psi = repeat_elem(upsample_psi, shape_x[2])

        y = Multiply()([upsample_psi, x])

        result = self.Conv(shape_x[2], 1, padding="same")(y)
        result_bn = BatchNormalization()(result)
        return result_bn


class AttResUNet(UNet):
    def convolution_block(
        self,
        input_tensor,
        n_filters: int,
        kernel_size: int = 3,
        batchnorm: bool = True,
        data_format: str = "channels_first",
        padding: str = "same",
        activation: str = "relu",
        kernel_initializer: str = "he_normal",
    ):
        # first layer
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        # Second layer.
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(x)
        if batchnorm:
            x = BatchNormalization()(x)

        # maybe a shortcut?
        # https://www.youtube.com/watch?v=L5iV5BHkMzM
        shortcut = self.Conv(n_filters, kernel_size=1, padding="same")(
            input_tensor
        )
        if batchnorm is True:
            shortcut = BatchNormalization()(shortcut)

        # Residual connection
        x = Add()([shortcut, x])
        x = Activation(activation)(x)
        return x

    def expansive_block(
        self,
        ci,
        cii,
        n_filters: int = 16,
        batchnorm: bool = True,
        dropout: float = 0.5,
        kernel_size: int = 3,
        strides: int = 2,
        data_format: str = "channels_first",
        activation: str = "relu",
        padding_style: str = "same",
    ):
        gating = self.gating_signal(ci, n_filters, True)
        att = self.attention_block(cii, gating, n_filters)

        u = self.ConvTranspose(
            n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding_style,
            data_format=data_format,
        )(ci)
        u = concatenate([u, att])
        u = self.SpatialDropout(dropout)(u)
        c = self.convolution_block(
            u,
            n_filters=n_filters,
            kernel_size=kernel_size,
            batchnorm=batchnorm,
            data_format=data_format,
            activation=activation,
            padding=padding_style,
        )
        return c

    def expanding_loop(
        self, contracted_layers, expansion_arguments: Dict[str, Any]
    ):
        list_c = [contracted_layers[-1]]
        iterator_expanded_blocks = range(self.number_of_conv_layers)
        iterator_contracted_blocks = reversed(iterator_expanded_blocks)
        n_filters = expansion_arguments["n_filters"]
        for i, c in zip(iterator_expanded_blocks, iterator_contracted_blocks):
            filter_expansion = 2 ** (c)
            expansion_arguments["n_filters"] = n_filters * filter_expansion

            c4 = self.expansive_block(
                list_c[i], contracted_layers[c], **expansion_arguments
            )
            list_c.append(c4)
        return c4


class ResUNet(UNet):
    def convolution_block(
        self,
        input_tensor,
        n_filters: int,
        kernel_size: int = 3,
        batchnorm: bool = True,
        data_format: str = "channels_first",
        padding: str = "same",
        activation: str = "relu",
        kernel_initializer: str = "he_normal",
    ):
        # first layer
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        # Second layer.
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(x)
        if batchnorm:
            x = BatchNormalization()(x)

        # maybe a shortcut?
        # https://www.youtube.com/watch?v=L5iV5BHkMzM
        shortcut = self.Conv(n_filters, kernel_size=1, padding="same")(
            input_tensor
        )
        if batchnorm is True:
            shortcut = BatchNormalization()(shortcut)
        # Residual connection

        x = Add()([shortcut, x])
        x = Activation(activation)(x)
        return x


class UNet1D(UNet):
    def __init__(
        self,
        **kwargs,
    ):
        self.Conv = Conv1D
        self.ConvTranspose = Conv1DTranspose
        self.SpatialDropout = Dropout
        self.MaxPooling = MaxPooling1D
        self.UpSampling = UpSampling1D
        kwargs["timesteps"] = 1
        self.data_format = kwargs["data_format"]
        super().__init__(**kwargs)

    @cached_property
    def model_input_shape(self):
        if self.data_format == "channels_first":
            return (self.num_bands, self.width)
        elif self.data_format == "channels_last":
            return (self.width, self.num_bands)

    def output_layer(self):
        outputDeep = self.deep_neural_network(
            n_filters=self.n_filters,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            data_format=self.data_format,
            activation_middle=self.activation_middle,
            kernel_size=self.kernel_size,
            padding=self.padding_style,
            # num_classes=self.num_classes,
        )
        outputDeep = self.Conv(
            24,
            self.kernel_size,
            activation=self.activation_end,
            data_format=self.data_format,
            padding=self.padding_style,
        )(outputDeep)

        new_shape = outputDeep.shape[1:]
        outputDeep = Reshape((new_shape[1], new_shape[0]))(outputDeep)

        outputDeep = self.Conv(
            self.num_classes,
            self.kernel_size,
            activation=self.activation_end,
            data_format=self.data_format,
            padding=self.padding_style,
        )(outputDeep)

        if self.padding > 0:
            outputDeep = Cropping2D(
                cropping=(
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                )
            )(outputDeep)
        self.output_layer = outputDeep
        return outputDeep


class AttResUNet1DBroad(AttResUNet):
    def __init__(
        self,
        **kwargs,
    ):
        self.Conv = Conv1D
        self.ConvTranspose = Conv1DTranspose
        self.SpatialDropout = Dropout
        self.MaxPooling = MaxPooling1D
        self.UpSampling = UpSampling1D

        kwargs["timesteps"] = 1
        self.data_format = kwargs["data_format"]
        super().__init__(**kwargs)

    @cached_property
    def model_input_shape(self):
        if self.data_format == "channels_first":
            return (self.num_bands, self.width)
        elif self.data_format == "channels_last":
            return (self.width, self.num_bands)

    def convolution_block(
        self,
        input_tensor,
        n_filters: int,
        kernel_size: int = 3,
        batchnorm: bool = True,
        data_format: str = "channels_first",
        padding: str = "same",
        activation: str = "relu",
        kernel_initializer: str = "he_normal",
    ):
        # first layer
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)

        paralel_convs = []
        for i in [3, 8, 12, 24, 48, 24 * 3, 24 * 7]:
            if i > self.width:
                continue
            # parale layer.
            xi = self.Conv(
                filters=n_filters,
                kernel_size=i,
                kernel_initializer=kernel_initializer,
                padding=padding,
                data_format=data_format,
                activation=activation,
            )(x)
            if batchnorm:
                xi = BatchNormalization()(xi)
            paralel_convs.append(xi)

        u = concatenate(paralel_convs)
        u = self.SpatialDropout(self.dropout)(u)

        # Second layer.
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(u)
        if batchnorm:
            x = BatchNormalization()(x)

        # maybe a shortcut?
        # https://www.youtube.com/watch?v=L5iV5BHkMzM
        shortcut = self.Conv(n_filters, kernel_size=1, padding="same")(
            input_tensor
        )
        if batchnorm is True:
            shortcut = BatchNormalization()(shortcut)
        # Residual connection

        x = Add()([shortcut, x])
        x = Activation(activation)(x)
        return x

    def output_layer(self):
        outputDeep = self.deep_neural_network(
            n_filters=self.n_filters,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            data_format=self.data_format,
            activation_middle=self.activation_middle,
            kernel_size=self.kernel_size,
            padding=self.padding_style,
            # num_classes=self.num_classes,
        )
        outputDeep = self.Conv(
            24,
            self.kernel_size,
            activation=self.activation_end,
            data_format=self.data_format,
            padding=self.padding_style,
        )(outputDeep)

        new_shape = outputDeep.shape[1:]
        outputDeep = Reshape((new_shape[1], new_shape[0]))(outputDeep)

        outputDeep = self.Conv(
            self.num_classes,
            self.kernel_size,
            activation=self.activation_end,
            data_format=self.data_format,
            padding=self.padding_style,
        )(outputDeep)

        if self.padding > 0:
            outputDeep = Cropping2D(
                cropping=(
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                )
            )(outputDeep)
        self.output_layer = outputDeep
        return outputDeep


class AttResUNet1D(AttResUNet):
    def __init__(
        self,
        **kwargs,
    ):
        self.Conv = Conv1D
        self.ConvTranspose = Conv1DTranspose
        self.SpatialDropout = Dropout
        self.MaxPooling = MaxPooling1D
        self.UpSampling = UpSampling1D

        kwargs["timesteps"] = 1
        self.data_format = kwargs["data_format"]
        super().__init__(**kwargs)

    @cached_property
    def model_input_shape(self):
        if self.data_format == "channels_first":
            return (self.num_bands, self.width)
        elif self.data_format == "channels_last":
            return (self.width, self.num_bands)

    def output_layer(self):
        outputDeep = self.deep_neural_network(
            n_filters=self.n_filters,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            data_format=self.data_format,
            activation_middle=self.activation_middle,
            kernel_size=self.kernel_size,
            padding=self.padding_style,
            # num_classes=self.num_classes,
        )
        outputDeep = self.Conv(
            24,
            self.kernel_size,
            activation=self.activation_end,
            data_format=self.data_format,
            padding=self.padding_style,
        )(outputDeep)

        new_shape = outputDeep.shape[1:]
        outputDeep = Reshape((new_shape[1], new_shape[0]))(outputDeep)

        outputDeep = self.Conv(
            self.num_classes,
            self.kernel_size,
            activation=self.activation_end,
            data_format=self.data_format,
            padding=self.padding_style,
        )(outputDeep)

        if self.padding > 0:
            outputDeep = Cropping2D(
                cropping=(
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                )
            )(outputDeep)
        self.output_layer = outputDeep
        return outputDeep
