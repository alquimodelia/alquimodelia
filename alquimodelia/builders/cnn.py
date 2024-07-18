from functools import cached_property
from typing import Any, Dict

import keras
from keras.layers import Activation, Add, Cropping2D, Multiply, concatenate
from keras.src.legacy.backend import int_shape

from alquimodelia.builders.base_builder import BaseBuilder
from alquimodelia.utils import count_number_divisions, repeat_elem



class CNN(BaseBuilder):
    """Base classe for Unet models."""

    def __init__(
        self,
        n_filters: int = 16,
        number_of_conv_layers: int = 0,
        kernel_size: int = 3,
        padding_style: str = "same",
        padding: int = 0,
        activation_middle: str = "relu",
        kernel_initializer: str = "he_normal",
        attention: bool = False,
        residual: bool = False,
        spatial_dropout: bool = True,
        classes_method: str = "Conv",
        cropping_method: str = "crop",
        pad_temp: bool = True,
        **kwargs,
    ):
        self._number_of_conv_layers = number_of_conv_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_style = padding_style
        self.spatial_dropout = spatial_dropout

        self.activation_middle = activation_middle
        self.kernel_initializer = kernel_initializer
        self.attention = attention
        self.residual = residual
        self.cropping_method=cropping_method
        # TODO: this variable is based on some shity assumtions
        # this variable is useless because this croping method is useless.
        self.pad_temp=pad_temp

        self.classes_method = classes_method  # Dense || Conv
        # TODO: study a way to make cropping within the convluition at the end, this way there is less pixels to actully calculate

        super().__init__(**kwargs)
