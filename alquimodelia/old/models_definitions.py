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
        "architecture_args": {
            "block_repetition": 2,
            "get_input_layer_args": {"time2vec_kernel_size": [1, 8]},
        },
    },
    "StackedDenseTime2Vec": {
        "arch": "DenseArch",
        "architecture_args": {
            "block_repetition": 2,
            "get_input_layer_args": {"time2vec_kernel_size": 24},
        },
    },
    "VanillaCNNTime2Vec": {
        "arch": "CNNArch",
        "architecture_args": {
            "get_input_layer_args": {"time2vec_kernel_size": 24}
        },
    },
    "VanillaDenseTime2Vec": {
        "arch": "DenseArch",
        "architecture_args": {
            "get_input_layer_args": {"time2vec_kernel_size": 24}
        },
    },
    "VanillaLSTMTime2Vec": {
        "arch": "LSTMArch",
        "architecture_args": {
            "get_input_layer_args": {"time2vec_kernel_size": 24}
        },
    },
    "StackedLSTMATime2Vec": {
        "arch": "LSTMArch",
        "architecture_args": {
            "block_repetition": 2,
            "get_input_layer_args": {"time2vec_kernel_size": 24},
        },
    },
    "EncoderDecoderTime2Vec": {"arch": "EncoderDecoder"},
    "TransformerTime2Vec": {"arch": "Transformer"},
    "StackedTransformerTime2Vec": {
        "arch": "Transformer",
        "architecture_args": {
            "block_repetition": 2,
            "get_input_layer_args": {"time2vec_kernel_size": 24},
        },
    },
    "Stacked6TransformerTime2Vec": {
        "arch": "Transformer",
        "architecture_args": {
            "block_repetition": 6,
            "get_input_layer_args": {"time2vec_kernel_size": 24},
        },
    },
    "StackedCNNTime2VecDist": {
        "arch": "CNNArch",
        "architecture_args": {
            "block_repetition": 2,
            "get_input_layer_args": {
                "time2vec_kernel_size": 24,
                "timedist": True,
            },
        },
    },
    "StackedDenseTime2VecDist": {
        "arch": "DenseArch",
        "architecture_args": {
            "block_repetition": 2,
            "get_input_layer_args": {
                "time2vec_kernel_size": 24,
                "timedist": True,
            },
        },
    },
    "VanillaCNNTime2VecDist": {
        "arch": "CNNArch",
        "architecture_args": {
            "get_input_layer_args": {
                "time2vec_kernel_size": 24,
                "timedist": True,
            }
        },
    },
    "VanillaDenseTime2VecDist": {
        "arch": "DenseArch",
        "architecture_args": {
            "get_input_layer_args": {
                "time2vec_kernel_size": 24,
                "timedist": True,
            }
        },
    },
    "VanillaLSTMTime2VecDist": {
        "arch": "LSTMArch",
        "architecture_args": {
            "get_input_layer_args": {
                "time2vec_kernel_size": 24,
                "timedist": True,
            }
        },
    },
    "StackedLSTMATime2VecDist": {
        "arch": "LSTMArch",
        "architecture_args": {
            "block_repetition": 2,
            "get_input_layer_args": {
                "time2vec_kernel_size": 24,
                "timedist": True,
            },
        },
    },
    "EncoderDecoderTime2VecDist": {"arch": "EncoderDecoder"},
    "TransformerTime2VecDist": {"arch": "Transformer"},
    "StackedTransformerTime2VecDist": {
        "arch": "Transformer",
        "architecture_args": {
            "block_repetition": 2,
            "get_input_layer_args": {
                "time2vec_kernel_size": 24,
                "timedist": True,
            },
        },
    },
    "Stacked6TransformerTime2VecDist": {
        "arch": "Transformer",
        "architecture_args": {
            "block_repetition": 6,
            "get_input_layer_args": {
                "time2vec_kernel_size": 24,
                "timedist": True,
            },
        },
    },
}


def get_model_from_def(
    model_name,
    architecture_args=None,
    input_args=None,
    model_structures=None,
    **kwargs
):
    from alquimodelia import archs

    model_structures = model_structures or MODELS_ARCHS
    architecture_args = architecture_args or {}
    input_args = input_args or {}

    # Get the model configuration
    model_conf = model_structures[model_name]
    architecture_args_conf = model_conf.get("architecture_args", {})
    architecture_args.update(architecture_args_conf)

    # Create the architecture and model
    architecture_class = getattr(archs, model_conf["arch"])
    forearch = architecture_class(**input_args)
    foremodel = forearch.architecture(**architecture_args)

    return foremodel
