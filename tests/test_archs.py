import keras
import numpy as np
import pytest

import alquimodelia

MODELS_ARCHS = {
    "StackedCNN": {
        "arch": "CNNArch",
        "architecture_args": {"block_repetition": 2},
    },
    "StackedDense": {
        "arch": "DenseArch",
        "architecture_args": {"block_repetition": 2},
    },
    "VanillaCNN": {
        "arch": "CNNArch",
    },
    "VanillaDense": {"arch": "DenseArch"},
    "VanillaLSTM": {"arch": "LSTMArch"},
    "StackedLSTMA": {
        "arch": "LSTMArch",
        "architecture_args": {"block_repetition": 2},
    },
    "UNET": {"arch": "UNETArch"},
    "EncoderDecoder": {"arch": "EncoderDecoder"},
    "Transformer": {"arch": "Transformer"},
    "StackedTransformer": {
        "arch": "Transformer",
        "architecture_args": {"block_repetition": 2},
    },
    "Stacked6Transformer": {
        "arch": "Transformer",
        "architecture_args": {"block_repetition": 6},
    },
    "StackedCNNTime2Vec": {
        "arch": "CNNArch",
        "architecture_args": {"block_repetition": 2,"get_input_layer_args":{"time2vec_kernel_size":[1,8]}},
    },
    "StackedDenseTime2Vec": {
        "arch": "DenseArch",
        "architecture_args": {"block_repetition": 2,"get_input_layer_args":{"time2vec_kernel_size":24}},
    },
    "VanillaCNNTime2Vec": {
        "arch": "CNNArch",
        "architecture_args": {"get_input_layer_args":{"time2vec_kernel_size":24}},

    },
    "VanillaDenseTime2Vec": {"arch": "DenseArch",
            "architecture_args": {"get_input_layer_args":{"time2vec_kernel_size":24}},

    },
    "VanillaLSTMTime2Vec": {"arch": "LSTMArch",
            "architecture_args": {"get_input_layer_args":{"time2vec_kernel_size":24}},
    },
    "StackedLSTMATime2Vec": {
        "arch": "LSTMArch",
        "architecture_args": {"block_repetition": 2,"get_input_layer_args":{"time2vec_kernel_size":24}},
    },
    "EncoderDecoderTime2Vec": {"arch": "EncoderDecoder"},
    "TransformerTime2Vec": {"arch": "Transformer"},
    "StackedTransformerTime2Vec": {
        "arch": "Transformer",
        "architecture_args": {"block_repetition": 2,"get_input_layer_args":{"time2vec_kernel_size":24}},
    },
    "Stacked6TransformerTime2Vec": {
        "arch": "Transformer",
        "architecture_args": {"block_repetition": 6,"get_input_layer_args":{"time2vec_kernel_size":24}},
    },

    "StackedCNNTime2VecDist": {
        "arch": "CNNArch",
        "architecture_args": {"block_repetition": 2,"get_input_layer_args":{"time2vec_kernel_size":24,"timedist":True}},
    },
    "StackedDenseTime2VecDist": {
        "arch": "DenseArch",
        "architecture_args": {"block_repetition": 2,"get_input_layer_args":{"time2vec_kernel_size":24,"timedist":True}},
    },
    "VanillaCNNTime2VecDist": {
        "arch": "CNNArch",
        "architecture_args": {"get_input_layer_args":{"time2vec_kernel_size":24,"timedist":True}},

    },
    "VanillaDenseTime2VecDist": {"arch": "DenseArch",
            "architecture_args": {"get_input_layer_args":{"time2vec_kernel_size":24,"timedist":True}},

    },
    "VanillaLSTMTime2VecDist": {"arch": "LSTMArch",
            "architecture_args": {"get_input_layer_args":{"time2vec_kernel_size":24,"timedist":True}},
    },
    "StackedLSTMATime2VecDist": {
        "arch": "LSTMArch",
        "architecture_args": {"block_repetition": 2,"get_input_layer_args":{"time2vec_kernel_size":24,"timedist":True}},
    },
    "EncoderDecoderTime2VecDist": {"arch": "EncoderDecoder"},
    "TransformerTime2VecDist": {"arch": "Transformer"},
    "StackedTransformerTime2VecDist": {
        "arch": "Transformer",
        "architecture_args": {"block_repetition": 2,"get_input_layer_args":{"time2vec_kernel_size":24,"timedist":True}},
    },
    "Stacked6TransformerTime2VecDist": {
        "arch": "Transformer",
        "architecture_args": {"block_repetition": 6,"get_input_layer_args":{"time2vec_kernel_size":24,"timedist":True}},
    },
}
# The model structures
model_structures = {
    "VanillaCNN": {"arch": "CNNArch"},
    "VanillaDense": {"arch": "DenseArch"},
    "VanillaLSTM": {"arch": "LSTMArch"},
    "StackedCNN": {
        "arch": "CNNArch",
        "architecture_args": {"block_repetition": 2},
    },
    "StackedDense": {
        "arch": "DenseArch",
        "architecture_args": {"block_repetition": 2},
    },
    "StackedLSTM": {
        "arch": "LSTMArch",
        "architecture_args": {"block_repetition": 2},
    },
    "UNET": {"arch": "UNETArch"},
    "EncoderDecoder": {"arch": "EncoderDecoder"},
    "Transformer": {"arch": "Transformer"},
    "StackedTransformer": {
        "arch": "Transformer",
        "architecture_args": {"block_repetition": 2},
    },
}
model_structures=MODELS_ARCHS
# The input arguments
input_args = {
    "X_timeseries": 168,
    "Y_timeseries": 24,
    "n_features_train": 18,
    "n_features_predict": 1,
}

# The compile arguments
compile_args = {
    "optimizer": "adam",
    "loss": "mse",
}


# The fixture for model name
@pytest.fixture(params=model_structures.keys())
def model_name(request):
    return request.param


# The test function
def test_model(model_name):
    # Clear the session
    keras.backend.clear_session()

    # Get the model configuration
    model_conf = model_structures[model_name]
    architecture_args = model_conf.get("architecture_args", {})

    # Create the architecture and model
    architecture_class = getattr(alquimodelia.archs, model_conf["arch"])
    forearch = architecture_class(**input_args)
    foremodel = forearch.architecture(**architecture_args)
    foremodel.summary()
    foremodel.compile(**compile_args)

    # Create the test data
    input_shape = (2, *foremodel.input_shape[1:])
    output_shape = (2, *foremodel.output_shape[1:])
    X_test = np.ones(input_shape)
    Y_test = np.ones(output_shape)

    # Fit the model
    foremodel.fit(X_test, Y_test)
