import os

os.environ["KERAS_BACKEND"] = "torch"
from inspect import signature

import keras
import numpy as np
from keras import layers, ops
from keras.layers import Add, Concatenate, Layer

from alquimodelia.builders.base_builder import BaseBuilder, SequenceBuilder
from alquimodelia.builders.cnn import CNN
from alquimodelia.builders.fcnn import FCNN

# TODO: it all
# This class should be able to build the 3 archs in: https://keras.io/examples/vision/image_classification_using_global_context_vision_transformer/
# ViT
# SwingTransformer
# GCVit


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[-3]
        width = input_shape[-2]
        channels = input_shape[-1]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class TubeletEmbedding(layers.Layer):
    # BUG: if the patch size is smaller then one dimension it goes to zero
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )

    def call(self, videos):
        projected_patches = self.projection(videos)
        return projected_patches


class PositionEmbedding(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, tokens):
        num_patches = ops.shape(tokens)[1]
        projection_dim = ops.shape(tokens)[-1]
        positions = ops.expand_dims(
            ops.arange(start=0, stop=num_patches, step=1), axis=0
        )
        positions_embeded = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(positions)
        if len(positions_embeded.shape) != len(tokens.shape):
            num_missing_dims = len(tokens.shape) - len(positions_embeded.shape)
            axis_to_expand = [-i for i in range(2, 2 + num_missing_dims)]
            positions_embeded = ops.expand_dims(
                positions_embeded, axis=axis_to_expand
            )

        return positions_embeded


def get_valid_parameters_of_class(class_name):
    all_signatures = []
    current_class = class_name
    while True:
        sig = signature(current_class.__init__)
        all_signatures.append(sig)

        parent_classes = current_class.__bases__
        if not parent_classes:
            break

        current_class = parent_classes[0]

    # Combine all signatures into a single dictionary
    combined_params = {}
    for s in all_signatures:
        for param in s.parameters.values():
            if param.name not in combined_params:
                combined_params[param.name] = param.default
    return combined_params


class Transformer(SequenceBuilder):
    def __init__(
        self,
        projection_dim: int = None,
        num_tokens_from_input: int = None,
        join_token_position: Layer = None,
        num_transformer_layers: int = None,
        num_heads: int = 4,
        transformer_units: list = None,
        tokenization_method=None,
        vector_tokens: bool = True,
        patch_size: int = None,
        use_embedding: bool = None,
        # TODO: change this interpretation, now I will feccth a cnn or a fcnn
        interpretation_method=None,
        # filters:int=None,
        **kwargs,
    ):
        self.num_tokens_from_input = num_tokens_from_input
        self.projection_dim = (
            projection_dim or num_tokens_from_input  # or filters
        )  # TODO: this should be taken out of inputshape if none of those exist
        if isinstance(join_token_position, str):
            join_token_position = getattr(layers, join_token_position)
        self.join_token_position = join_token_position or Add()
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.tokenization_method = tokenization_method
        self.patch_size = patch_size
        self.use_embedding = use_embedding

        # or [
        #     self.projection_dim * 2,
        #     self.projection_dim,
        # ]
        self.vector_tokens = vector_tokens
        self.interpretation_method = interpretation_method or CNN
        super().__init__(**kwargs)

    def get_input_layer(self):
        # This is to get the layer to enter the arch, here you can add augmentation or other processing

        # TODO: add pathches here as an option
        return self.input_layer

    def tokenization(self, input_layer):
        # Tokenize method for the input
        tokens = input_layer
        # TODO: this cant be here:
        if self.projection_dim is None:
            self.projection_dim = input_layer.shape[-1]

            self.transformer_units = [
                self.projection_dim * 2,
                self.projection_dim,
            ]
        # TODO: tokenization methods, conv3d for video, embeding for text, dense for images

        use_embedding = True
        if self.patch_size is not None:
            # TODO: rethink this to make more dinamic now it is force tubelet for video and viT for image
            if len(self.model_input_shape) == 4:
                tokens = TubeletEmbedding(
                    embed_dim=self.projection_dim, patch_size=self.patch_size
                )(tokens)
            if len(self.model_input_shape) == 3:
                tokens = Patches(self.patch_size)(tokens)
                tokens = layers.Dense(self.projection_dim)(tokens)
        else:
            # Lets assume timeseries have the x_timeseries and channels
            if len(self.model_input_shape) == 2:
                if self.use_embedding is None:
                    use_embedding = False
            # Lets assume text just have the channels
            if len(self.model_input_shape) == 1:
                vocab_size = self.model_input_shape[-1]
                tokens = layers.Embedding(
                    input_dim=vocab_size, output_dim=self.projection_dim
                )(tokens)
        if self.use_embedding is None:
            self.use_embedding = use_embedding

        # So there is usually an initial tranfomation of tokens to a representation, and this is representation number is also used as the embeding number

        if self.vector_tokens:
            if len(tokens.shape) > 3:
                channel_num = tokens.shape[-1]
                shape_tokens = ops.prod(tokens.shape[1:-1])
                tokens = layers.Reshape((shape_tokens, channel_num))(tokens)

        return tokens

    def embedding(self, tokens):
        if self.use_embedding is False:
            return tokens

        positions_embeded = PositionEmbedding()(tokens)
        # embedding_tokens = self.join_token_position(
        #     [tokens, positions_embeded]
        # )
        embedding_tokens = tokens + positions_embeded
        # embedding_tokens = Add()([tokens,positions_embeded])

        return embedding_tokens

    def mlp(self, enconded_tokens, hidden_units=None, dropout_rate=None):
        hidden_units = hidden_units or self.transformer_units
        dropout_rate = dropout_rate or self.dropout_rate
        for units in hidden_units:
            enconded_tokens = layers.Dense(
                units, activation=keras.activations.gelu
            )(enconded_tokens)
            enconded_tokens = layers.Dropout(dropout_rate)(enconded_tokens)
        return enconded_tokens

    def transformer_block(self, tokens):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(tokens)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, tokens])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x4 = self.mlp(
            x3, hidden_units=self.transformer_units, dropout_rate=0.1
        )
        # Skip connection 2.

        # NOTE: I have seen this add with x3 and with x4 (before and after normalization)
        tokens = layers.Add()([x4, x2])

        return tokens

    def encoder_block(self, tokens):
        # Create multiple layers of the Transformer block.
        for _ in range(self.num_transformer_layers):
            tokens = self.transformer_block(tokens)
        representation = layers.LayerNormalization(epsilon=1e-6)(tokens)
        representation = layers.Dropout(self.dropout_rate)(representation)
        return representation

    def define_model(self):
        input_layer = self.get_input_layer()
        tokens = self.tokenization(input_layer)
        embedding_tokens = self.embedding(tokens)
        # On a straight out transformer this is the Attention block
        encoded_tokens = self.encoder_block(embedding_tokens)
        # encoded_tokens = self.match_layer_to_output(encoded_tokens)

        self.last_arch_layer = encoded_tokens
        return encoded_tokens

    def define_output_layer(self):
        encoded_tokens = self.last_arch_layer
        original_args = self.__dict__
        inter_args = get_valid_parameters_of_class(self.interpretation_method)
        original_args = {
            k: v for k, v in original_args.items() if k in inter_args.keys()
        }
        original_args = {
            k: v for k, v in original_args.items() if v is not None
        }
        original_args = {
            k: v
            for k, v in original_args.items()
            if not isinstance(v, keras.KerasTensor)
        }

        interpretation_model = self.interpretation_method(
            input_layer=encoded_tokens, **original_args
        )

        self.output_layer = interpretation_model.output_layer

    def match_layer_to_output(self, output_layer):

        out_shape_diff = (
            len(self.model_output_shape) + 1 - len(output_layer.shape)
        )
        if len(output_layer.shape) < len(self.model_output_shape) + 1:
            axis_to_expand = [-i for i in range(1, 1 + out_shape_diff)]
            output_layer = ops.expand_dims(output_layer, axis=axis_to_expand)

            h_out = self.dict_dimension_to_predict["H"]
            w_out = self.dict_dimension_to_predict["W"]

            h_in = output_layer.shape[1]
            w_in = output_layer.shape[2]

            num_pixels_out = ops.prod(self.model_output_shape)
            num_pixels_in = ops.prod(output_layer.shape[1:])

            h_diff = h_in - h_out
            w_diff = w_in - w_out
            # if num_pixels_in<num_pixels_out:

            #     ratio_pixel = num_pixels_out/num_pixels_in
            #     h_reshape = h_out/ratio_pixel
            #     w_reshape = w_out/ratio_pixel

            if h_diff > 0:
                output_layer = layers.Conv2D(
                    self.interpretation_filters,
                    kernel_size=(abs(h_diff) + 1, 1),
                    strides=1,
                    padding="valid",
                )(output_layer)
            if w_diff > 0:
                output_layer = layers.Conv2D(
                    self.interpretation_filters,
                    kernel_size=(1, abs(w_diff) + 1),
                    strides=1,
                    padding="valid",
                )(output_layer)

            if h_diff < 0:
                output_layer = layers.Conv2DTranspose(
                    self.interpretation_filters,
                    kernel_size=(abs(h_diff) + 1, 1),
                    strides=1,
                    padding="valid",
                )(output_layer)
            if w_diff < 0:
                output_layer = layers.Conv2DTranspose(
                    self.interpretation_filters,
                    kernel_size=(1, abs(w_diff) + 1),
                    strides=1,
                    padding="valid",
                )(output_layer)

        # TODO: make this interpretation layer something like the CNN
        # output_layer=self.interpretation_layer(output_layer)
        # output_layer = self.mlp(output_layer, hidden_units=[self.num_classes*2,self.num_classes])

        return output_layer

    def model_setup(self):
        # Any needed setup before building and conecting the layers
        self.num_transformer_layers = self.num_transformer_layers or self.num_sequences
        # make path_size zero if is not an image.
        if self.patch_size is not None:
            if len(self.model_input_shape) < 3:
                self.patch_size = None

        # TODO: define whats comming for output
        # self.flatten_output=True


# input_args = {
#     "x_timesteps": 168,  # Number of sentinel images
#     "y_timesteps": 24,  # Number of volume maps
#     "num_features_to_train": 17,  # Number of sentinel bands
#     "num_classes": 1,  # We just want to predict the volume linearly
#     "height":1,
#     "width":1,
#     "num_tokens_from_input":None,
#     "vector_tokens":True,
#     "num_transformer_layers":1,
#     "patch_size":12,
#     "interpretation_filters":200,
# }

# # input_args = {
# #     "x_timesteps": 8,  # Number of sentinel images
# #     "y_timesteps": 1,  # Number of volume maps
# #     "num_features_to_train": 12,  # Number of sentinel bands
# #     "num_classes": 1,  # We just want to predict the volume linearly
# #     "height":128,
# #     "width":128,
# #     "num_tokens_from_input":None,
# #     "vector_tokens":True,
# #     "num_transformer_layers":1,
# #     "patch_size":8,
# #     "interpretation_filters":200,

# # }

# # cnn = CNN(**input_args)
# transformer = Transformer(#model_arch="transformer",
#                           **input_args)
# transformer.model.summary()
# print("sssss")
