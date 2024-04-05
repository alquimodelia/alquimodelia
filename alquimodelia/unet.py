from functools import cached_property
from typing import Any, Dict

import keras
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Cropping2D,
    Multiply,
    concatenate,
)
from keras.src.legacy.backend import int_shape

from alquimodelia.alquimodelia import ModelMagia
from alquimodelia.utils import count_number_divisions, repeat_elem


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
        attention: bool = False,
        residual: bool = False,
        dimensions_to_use=None,
        spatial_dropout: bool = True,
        classes_method: str = "Conv",
        **kwargs,
    ):
        self._number_of_conv_layers = number_of_conv_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batchnorm = batchnorm
        self.padding = padding
        self.padding_style = padding_style
        self.dropout = dropout
        self.spatial_dropout = spatial_dropout

        self.activation_middle = activation_middle
        self.activation_end = activation_end
        self.kernel_initializer = kernel_initializer
        self.attention = attention
        self.residual = residual

        self.dimensions_to_use = dimensions_to_use  # or ("T", "H", "W", "B")
        self._dimensions_to_use = self.dimensions_to_use

        self.classes_method = classes_method  # Dense || Conv

        super().__init__(**kwargs)

    def model_setup(self):
        self.Conv = getattr(keras.layers, f"Conv{self.conv_dimension}D")
        self.ConvTranspose = getattr(
            keras.layers, f"Conv{self.conv_dimension}DTranspose"
        )
        self.MaxPooling = getattr(
            keras.layers, f"MaxPooling{self.conv_dimension}D"
        )
        self.UpSampling = getattr(
            keras.layers, f"UpSampling{self.conv_dimension}D"
        )
        if self.spatial_dropout:
            self.Dropout = getattr(
                keras.layers, f"SpatialDropout{self.conv_dimension}D"
            )
        else:
            self.Dropout = keras.layers.Dropout

    def get_input_layer(self):
        return self.input_layer

    @cached_property
    def model_input_shape(self):
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
            input_shape = [f for f in self.input_dimensions if f > 1]
            if self.channels_dimension == 0:
                input_shape.insert(0, self.num_features_to_train)
            else:
                input_shape.append(self.num_features_to_train)
        # TODO: create the dimension_to_use atribute with the corret order

        return input_shape

    @cached_property
    def conv_dimension(self):
        # 1D, 2D, or 3D convulutions
        return len(self.model_input_shape) - 1

    @cached_property
    def number_of_conv_layers(self):
        if self._number_of_conv_layers == 0:
            number_of_layers = []
            study_shape = list(self.model_input_shape)
            study_shape.pop(self.channels_dimension)
            study_shape = tuple(study_shape)
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

    def residual_block(
        self,
        input_tensor,
        x,
        n_filters: int,
        batchnorm: bool = True,
        activation: str = "relu",
    ):
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
        residual: bool = False,
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

        if residual:
            x = self.residual_block(
                input_tensor, x, n_filters, batchnorm, activation
            )
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
        residual: bool = False,
    ):
        c1 = self.convolution_block(
            input_img,
            n_filters=n_filters,
            kernel_size=kernel_size,
            batchnorm=batchnorm,
            data_format=data_format,
            activation=activation,
            padding=padding,
            residual=residual,
        )
        p1 = self.MaxPooling(strides, padding=padding)(c1)
        p1 = self.Dropout(dropout)(p1)
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
        attention: bool = False,
    ):
        if attention:
            gating = self.gating_signal(ci, n_filters, True)
            cii = self.attention_block(cii, gating, n_filters)

        u = self.ConvTranspose(
            n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding_style,
            data_format=data_format,
        )(ci)
        u = concatenate([u, cii])
        u = self.Dropout(dropout)(u)
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
        residual: bool = False,
        attention: bool = False,
    ):
        """Build deep neural network."""
        input_img = self.get_input_layer()
        # self.define_number_convolution_layers()

        contracting_arguments = {
            "n_filters": n_filters,
            "batchnorm": batchnorm,
            "dropout": dropout,
            "kernel_size": kernel_size,
            "padding": padding,
            "data_format": data_format,
            "activation": activation_middle,
            "residual": residual,
        }
        expansion_arguments = {
            "n_filters": n_filters,
            "batchnorm": batchnorm,
            "dropout": dropout,
            "data_format": data_format,
            "activation": activation_middle,
            "kernel_size": kernel_size,
            "attention": attention,
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
        # TODO: fix for multiple dimensions
        sss = (shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2])
        # Upsampling here only acounts for a whole division,
        # and with all dimension having that diference
        upsample_psi = self.UpSampling(size=sss[0])(sigmoid_xg)
        # If its only only there is not need to repeat the tensor, and the multiply will do this
        if upsample_psi.shape[-1] != 1:
            last_dim_ratio = int(x.shape[-1] / upsample_psi.shape[-1])
            upsample_psi = repeat_elem(upsample_psi, last_dim_ratio)

        y = Multiply()([upsample_psi, x])

        result = self.Conv(shape_x[2], 1, padding="same")(y)
        result_bn = BatchNormalization()(result)
        return result_bn

    def classes_collapse(self, outputDeep):
        if self.classes_method == "Conv":
            outputDeep = self.Conv(
                self.num_classes,
                self.kernel_size,
                activation=self.activation_end,
                data_format=self.data_format,
                padding=self.padding_style,
            )(outputDeep)
        elif self.classes_method == "Dense":
            outputDeep = keras.layers.Dense(
                units=self.num_classes, activation=self.activation_end
            )(outputDeep)
        return outputDeep

    def define_output_layer(self):
        # Output of the neural network
        outputDeep = self.deep_neural_network(
            n_filters=self.n_filters,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            data_format=self.data_format,
            activation_middle=self.activation_middle,
            kernel_size=self.kernel_size,
            padding=self.padding_style,
            attention=self.attention,
            residual=self.residual,
        )

        # "Time" dimension colapse (or expansion)
        if self.y_timesteps < self.x_timesteps:
            # TODO: the channels 2st wont work training on CPU
            # TODO: this might not work on all 1D, 2D...
            # TODO: a transpose or reshape might be a better alternative if no GPU is available
            # On torch it seems to work on CPU.
            outputDeep = self.Conv(
                self.y_timesteps,
                self.kernel_size,
                activation=self.activation_end,
                data_format=self.opposite_data_format(),
                padding=self.padding_style,
            )(outputDeep)
        # outputDeep = ops.transpose(outputDeep, axes=[0,4,2,3,1])

        # new_shape = outputDeep.shape[1:]
        # outputDeep = Reshape((new_shape[1], new_shape[0]))(outputDeep)

        # Classes colapse (or expansion)
        outputDeep = self.classes_collapse(outputDeep)

        # TODO: croping should be in a different function to treat output
        if self.padding > 0:
            outputDeep = Cropping2D(
                cropping=(
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                )
            )(outputDeep)
        self.output_layer = outputDeep
        return outputDeep


class AttResUNet(UNet):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs, attention=True, residual=True)


class ResUNet(UNet):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs, residual=True)
