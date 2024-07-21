from functools import cached_property

import keras

from alquimodelia.builders.base_builder import SequenceBuilder
from alquimodelia.utils import count_number_divisions

# TODO: recheck everything about filters and stuffy


class CNN(SequenceBuilder):
    """Base classe for CNN models."""

    def __init__(
        self,
        number_of_conv_layers: int = 0,
        spatial_dropout: bool = False,
        padding: int = 0,
        classes_method: str = "Dense",
        **kwargs,
    ):
        self.spatial_dropout = spatial_dropout
        self._number_of_conv_layers = number_of_conv_layers
        self.padding = padding
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
        self.Cropping = getattr(
            keras.layers, f"Cropping{self.conv_dimension}D"
        )
        # TODO: find way to do croping in all dimensions
        if self.conv_dimension == 2:
            self.cropping_tuple = (
                (self.padding, self.padding),
                (self.padding, self.padding),
            )
        elif self.conv_dimension == 3:
            self.cropping_tuple = (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            )
        self.num_sequences = self.num_sequences or self.number_of_conv_layers

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

    @cached_property
    def conv_dimension(self):
        # 1D, 2D, or 3D convulutions
        return len(self.model_input_shape) - 1

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
        x = self.Dropout(self.dropout_rate)(x)

        return x
