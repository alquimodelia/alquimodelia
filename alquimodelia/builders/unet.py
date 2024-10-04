from typing import Any, Dict

import keras
from keras.layers import Activation, Add, Multiply, concatenate
from keras.src.legacy.backend import int_shape

from alquimodelia.builders.cnn import CNN
from alquimodelia.utils import repeat_elem


class UNet(CNN):
    """Base classe for Unet models."""

    def __init__(
        self,
        n_filters: int = 16,
        kernel_size: int = 3,
        padding_style: str = "same",
        activation_middle: str = "relu",
        kernel_initializer: str = "he_normal",
        attention: bool = False,
        residual: bool = False,
        cropping_method: str = "crop",
        pad_temp: bool = True,
        spatial_dropout: bool = True,
        double_interpretation: bool = True,
        **kwargs,
    ):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding_style = padding_style

        self.activation_middle = activation_middle
        self.kernel_initializer = kernel_initializer
        self.attention = attention
        self.residual = residual
        self.cropping_method = cropping_method
        # TODO: this variable is based on some shity assumtions
        # this variable is useless because this croping method is useless.
        self.pad_temp = pad_temp
        self.extra_crop=None

        # TODO: study a way to make cropping within the convluition at the end, this way there is less pixels to actully calculate

        super().__init__(
            spatial_dropout=spatial_dropout,
            double_interpretation=double_interpretation,
            flatten_output=False,
            **kwargs,
        )

    def residual_block(
        self,
        input_tensor,
        x,
        n_filters: int,
        normalization: bool = True,
        activation: str = "relu",
    ):
        # maybe a shortcut?
        # https://www.youtube.com/watch?v=L5iV5BHkMzM
        shortcut = self.Conv(n_filters, kernel_size=1, padding="same")(
            input_tensor
        )
        if normalization is True:
            shortcut = self.normalization()(shortcut)

        # Residual connection
        x = Add()([shortcut, x])
        x = Activation(activation)(x)
        return x

    def convolution_block(
        self,
        input_tensor,
        n_filters: int,
        kernel_size: int = 3,
        normalization: bool = True,
        data_format: str = "channels_first",
        padding: str = "same",
        activation: str = "relu",
        kernel_initializer: str = "he_normal",
        residual: bool = False,
        kernel_size_conv_one=None,
        kernel_size_conv_two=None,
        padding_conv_one=None,
        padding_conv_two=None,
        n_filters_one=None,
        n_filters_two=None,
    ):
        kernel_size_conv_one = kernel_size_conv_one or kernel_size
        kernel_size_conv_two = kernel_size_conv_two or kernel_size
        padding_conv_one = padding_conv_one or padding
        padding_conv_two = padding_conv_two or padding
        n_filters_one = n_filters_one or n_filters
        n_filters_two = n_filters_two or n_filters

        # first layer
        x = self.Conv(
            filters=n_filters_one,
            kernel_size=kernel_size_conv_one,
            kernel_initializer=kernel_initializer,
            padding=padding_conv_one,
            data_format=data_format,
            activation=activation,
        )(input_tensor)
        if normalization is not None:
            x = self.normalization()(x)
        # Second layer.
        x = self.Conv(
            filters=n_filters_two,
            kernel_size=kernel_size_conv_two,
            kernel_initializer=kernel_initializer,
            padding=padding_conv_two,
            data_format=data_format,
            activation=activation,
        )(x)
        if normalization is not None:
            x = self.normalization()(x)

        if residual:
            x = self.residual_block(
                input_tensor, x, n_filters, normalization, activation
            )
        return x

    def contracting_block(
        self,
        input_img,
        n_filters: int = 16,
        normalization: bool = True,
        dropout_rate: float = 0.25,
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
            normalization=normalization,
            data_format=data_format,
            activation=activation,
            padding=padding,
            residual=residual,
        )
        p1 = self.MaxPooling(strides, padding=padding)(c1)
        p1 = self.Dropout(dropout_rate)(p1)
        return p1, c1

    def expansive_block(
        self,
        ci,
        cii,
        n_filters: int = 16,
        normalization: bool = True,
        dropout_rate: float = 0.5,
        kernel_size: int = 3,
        strides: int = 2,
        data_format: str = "channels_first",
        activation: str = "relu",
        padding_style: str = "same",
        attention: bool = False,
        kernel_size_transpose=None,
        kernel_size_conv=None,
        **kwargs,
    ):
        if attention:
            gating = self.gating_signal(ci, n_filters, True)
            cii = self.attention_block(cii, gating, n_filters)

        kernel_size_transpose = kernel_size_transpose or kernel_size
        kernel_size_conv = kernel_size_conv or kernel_size

        u = self.ConvTranspose(
            n_filters,
            kernel_size=kernel_size_transpose,
            strides=strides,
            padding=padding_style,
            data_format=data_format,
        )(ci)
        u = concatenate([u, cii])
        u = self.Dropout(dropout_rate)(u)
        c = self.convolution_block(
            u,
            n_filters=n_filters,
            kernel_size=kernel_size_conv,
            normalization=normalization,
            data_format=data_format,
            activation=activation,
            padding=padding_style,
            **kwargs,
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
            contracting_arguments_to_use = {**contracting_arguments}
            if self.cropping_method == "expansion":
                if self.pad_temp is True:
                    temp_crop = contracting_arguments["kernel_size"]
                else:
                    temp_crop = 1
                # TODO: this is generating more neurons than just croping at the end, which kills the purpose
                if i == 0:
                    kernel_crop = self.padding + 1
                    contracting_arguments_to_use.update(
                        {
                            "padding": "valid",
                            "kernel_size": (
                                temp_crop,
                                kernel_crop,
                                kernel_crop,
                            ),
                        }
                    )
            p, c = self.contracting_block(
                old_p, **contracting_arguments_to_use
            )
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
        iteration_counter = 0
        for i, c in zip(iterator_expanded_blocks, iterator_contracted_blocks):
            iteration_counter += 1
            filter_expansion = 2 ** (c)
            expansion_arguments["n_filters"] = n_filters * filter_expansion
            iii_shape = list_c[i].shape
            ccc_shape = contracted_layers[c].shape
            expansion_arguments_to_use = {**expansion_arguments}
            if iteration_counter == self.number_of_conv_layers:
                if self.cropping_method == "contraction_final_4_2": # it is smooth, still not sure if plausable to keep
                    expansion_arguments_to_use.update(
                        {
                            "padding_conv_one": "valid",
                            "padding_conv_two": "valid",
                            "kernel_size_conv_one": (
                                1,
                                int((self.kernel_size + self.padding + 2) / 2),
                                int((self.kernel_size + self.padding + 2) / 2),
                            ),
                            "kernel_size_conv_two": (
                                1,
                                int((self.kernel_size + self.padding + 2) / 2),
                                int((self.kernel_size + self.padding + 2) / 2),
                            ),
                        }
                    )
                    self.extra_crop=1

                if self.cropping_method == "contraction_final_4": # it maintains the limit problem, remove
                    expansion_arguments_to_use.update(
                        {
                            "padding_conv_one": "valid",
                            "padding_conv_two": "valid",
                            "kernel_size_conv_one": (
                                1,
                                int((self.kernel_size + self.padding + 2) / 2)
                                + 1,
                                int((self.kernel_size + self.padding + 2) / 2)
                                + 1,
                            ),
                            "kernel_size_conv_two": (
                                1,
                                int((self.kernel_size + self.padding + 2) / 2)
                                + 1,
                                int((self.kernel_size + self.padding + 2) / 2)
                                + 1,
                            ),
                            # "n_filters_one":self.num_classes, #TODO: latest last shape
                            # "n_filters_two":self.num_classes, #TODO: latest last shape
                        }
                    )
            c4 = self.expansive_block(
                list_c[i], contracted_layers[c], **expansion_arguments_to_use
            )
            list_c.append(c4)
        return c4

    def deep_neural_network(
        self,
        input_img,
        n_filters: int = 16,
        dropout_rate: float = 0.2,
        normalization: bool = True,
        data_format: str = "channels_last",
        activation_middle: str = "relu",
        kernel_size: int = 3,
        padding: str = "same",
        residual: bool = False,
        attention: bool = False,
    ):
        """Build deep neural network."""

        contracting_arguments = {
            "n_filters": n_filters,
            "normalization": normalization,
            "dropout_rate": dropout_rate,
            "kernel_size": kernel_size,
            "padding": padding,
            "data_format": data_format,
            "activation": activation_middle,
            "residual": residual,
        }
        expansion_arguments = {
            "n_filters": n_filters,
            "normalization": normalization,
            "dropout_rate": dropout_rate,
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

        x = self.normalization()(x)
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
        result_bn = self.normalization()(result)
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
            outputDeep = self.interpretation_layer(outputDeep)
            # outputDeep = keras.layers.Dense(
            #     units=self.num_classes, activation=self.activation_end
            # )(outputDeep)
        return outputDeep

    def define_model(self):
        input_img = self.get_input_layer()
        # Output of the neural network
        outputDeep = self.deep_neural_network(
            input_img=input_img,
            n_filters=self.n_filters,
            dropout_rate=self.dropout_rate,
            normalization=self.normalization,
            data_format=self.data_format,
            activation_middle=self.activation_middle,
            kernel_size=self.kernel_size,
            padding=self.padding_style,
            attention=self.attention,
            residual=self.residual,
        )
        self.last_arch_layer = outputDeep
        return outputDeep

    def define_output_layer(self):
        outputDeep = self.last_arch_layer
        # "Time" dimension colapse (or expansion)
        if self.y_timesteps < self.x_timesteps:
            # TODO: this might not work on all 1D, 2D...
            # TODO: a transpose or reshape might be a better alternative if no GPU is available
            # On torch it seems to work on CPU.
            if self.cropping_method == "contraction_final": # it maintains the limit problem, remove
                outputDeep = self.Conv(
                    self.y_timesteps,
                    self.kernel_size + self.padding + 2,
                    activation=self.activation_end,
                    data_format=self.opposite_data_format(),
                    padding="valid",
                )(outputDeep)
            if self.cropping_method == "contraction_final_1_2": # it is smooth, still not sure if plausable to keep
                outputDeep = self.Conv(
                    self.y_timesteps,
                    self.kernel_size + self.padding,
                    activation=self.activation_end,
                    data_format=self.opposite_data_format(),
                    padding="valid",
                )(outputDeep)
                self.extra_crop=1
            if self.cropping_method == "contraction_final_2": # Not good remove on future implementation
                outputDeep = self.Conv(
                    self.y_timesteps,
                    (
                        self.kernel_size + self.padding + 2,
                        self.kernel_size + self.padding + 2,
                        1,
                    ),
                    activation=self.activation_end,
                    data_format=self.opposite_data_format(),
                    padding="valid",
                )(outputDeep)

            else:
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
            if self.cropping_method == "crop":
                outputDeep = self.Cropping(cropping=self.cropping_tuple)(
                    outputDeep
                )
        if self.extra_crop:
            outputDeep = self.Cropping(cropping=((0,0),(self.extra_crop,self.extra_crop),(self.extra_crop,self.extra_crop)))(
                outputDeep
            )

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
