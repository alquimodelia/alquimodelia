import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, ops
from keras.layers import Add, Concatenate, Layer

from alquimodelia.builders.base_builder import BaseBuilder

# TODO: it all
# This class should be able to build the 3 archs in: https://keras.io/examples/vision/image_classification_using_global_context_vision_transformer/
# ViT
# SwingTransformer
# GCVit


class Transformer(BaseBuilder):
    def __init__(
        self,
        projection_dim: int = None,
        initial_representation_dim: int = None,
        join_token_position: Layer = None,
        num_transformer_layers: int = 6,
        num_heads: int = 4,
        transformer_units: list = None,
        **kwargs,
    ):
        self.initial_representation_dim = initial_representation_dim
        self.projection_dim = (
            projection_dim or initial_representation_dim
        )  # TODO: this should be taken out of inputshape if none of those exist
        if isinstance(join_token_position, str):
            join_token_position = getattr(layers, join_token_position)
        self.join_token_position = join_token_position or Add()
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.transformer_units = transformer_units or [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        super().__init__(**kwargs)

    def get_input_layer(self):
        # This is to get the layer to enter the arch, here you can add augmentation or other processing

        # TODO: add pathches here as an option
        return self.input_layer

    def embedding_method(self, tokens):
        # TODO: create Embedding
        # TODO: this should be the number of tokens, and not the number of batches!
        num_patches = ops.shape(tokens)[1]
        projection_dim = self.projection_dim or ops.shape(tokens)[-1]
        positions = ops.arange(start=0, stop=num_patches)
        positions_embeded = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(positions)

        return positions_embeded

    def tokenization(self, input_layer):
        # Tokenize method for the input
        tokens = input_layer
        # Do we pass the tokens to a embedding layer?
        # There tokens are passed trhoufg a dense layer to diminuir dimnesionalidade

        # So there is usually an initial tranfomation of tokens to a representation, and this is representation number is also used as the embeding number
        if self.initial_representation_dim:
            tokens = layers.Dense(self.initial_representation_dim)(tokens)
        position_embedding = self.embedding_method(tokens)

        embedding_tokens = self.join_token_position(
            [tokens, position_embedding]
        )

        return embedding_tokens

    def mlp(self, enconded_tokens, hidden_units=None, dropout_rate=None):
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
        representation = layers.Dropout(0.5)(representation)
        return representation

    def define_model(self):
        input_layer = self.get_input_layer()
        tokens = self.tokenization(input_layer)

        # On a straight out transformer this is the Attention block
        encoded_tokens = self.encoder_block(tokens)

        self.last_arch_layer = encoded_tokens
        return encoded_tokens

    def define_output_layer(self):
        # This should only deal with the last layer, it can be used to define the classification for instance, or multiple methods to close the model
        # it should use the last_arch_layer
        # it should define self.output_layer
        enconded_tokens = self.last_arch_layer
        output_layer = self.mlp(enconded_tokens)
        self.output_layer = output_layer
        return output_layer

    def model_setup(self):
        # Any needed setup before building and conecting the layers
        pass


# input_args = {
#     "x_timesteps": 8,  # Number of sentinel images
#     "y_timesteps": 1,  # Number of volume maps
#     "num_features_to_train": 12,  # Number of sentinel bands
#     "num_classes": 1,  # We just want to predict the volume linearly
#     "height":128,
#     "width":1,
# }

# transformer = Transformer(#model_arch="tranformer",
#                           **input_args)
# transformer.model.summary()
# print("sssss")
