import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
from keras import layers, ops
from keras.layers import Add, Concatenate, Layer

from alquimodelia.builders.base_builder import BaseBuilder


def create_vit_segmentation(num_classes):
    input_shape = (224, 224, 3)  # Adjust this based on your desired input size
    
    inputs = keras.Input(shape=input_shape)
    
    # Augment data.
    
    # Create patches.
    patch_size=16
    projection_dim=3
    num_patches=196

    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)


    # Create multiple layers of the Transformer block.
    for _ in range(12):  # Adjust based on your needs
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp = layers.Dense(projection_dim, activation='relu')(x3)
        x3 = layers.Dropout(0.1)(mlp)
        x3 = layers.Add()([x3, x2])
        encoded_patches = layers.LayerNormalization(epsilon=1e-6)(x3)

    # Create a [batch_size, num_patches, projection_dim] tensor.
    representation = layers.GlobalAveragePooling1D()(encoded_patches)
    
    # Add MLP.
    features = layers.Dense(1024, activation='relu')(representation)
    features = layers.Dropout(0.5)(features)
    
    # Upsample to original image resolution
    outputs = layers.Conv2DTranspose(num_classes, kernel_size=2, strides=(2, 2), padding='same')(features)
    
    # Apply softmax activation
    outputs = layers.Activation('softmax')(outputs)
    
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the segmentation model
segmentation_model = create_vit_segmentation(num_classes=10)