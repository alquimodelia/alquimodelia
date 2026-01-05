

from alquimodelia.builders.unet import UNet


class ResNet(UNet):
    def __init__(
        self,
        break_image_dims_with_conv=False,
        residual=True,
        **kwargs,
    ):
        super().__init__(
            residual=residual,
            break_image_dims_with_conv=break_image_dims_with_conv, 
                         **kwargs)



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
        # expansion_arguments = {
        #     "n_filters": n_filters,
        #     "normalization": normalization,
        #     "dropout_rate": dropout_rate,
        #     "data_format": data_format,
        #     "activation": activation_middle,
        #     "kernel_size": kernel_size,
        #     "attention": attention,
        # }

        contracted_layers = self.contracting_loop(
            input_img, contracting_arguments
        )
        # unet_output = self.expanding_loop(
        #     contracted_layers, expansion_arguments
        # )

        return contracted_layers[-1]