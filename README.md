# Alquimodelia

Alquimodelia is a Python package that provides a Keras-based forecast model builder.

[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/keras-2.4.3-blue)](https://keras.io/)

It provides the arquitectures for CNN, LSTM, and Encoder Decoder, and even from imagery UNET.
Any suggestions and tips are welcome.
Use this to fastly have your forecast models ready to use!


## Usage

To use Alquimodelia, follow these steps:

```bash
    pip install alquimodelia
```

Since Alquimodelia is based on keras-core you can choose which backend to use, otherwise it will default to tensorflow.
To change backend change the ```KERAS-BACKEND``` enviromental variable. Follow [this](https://keras.io/keras/#configuring-your-backend).

To get an arquiteture you only need to have a simple configuration and call the module:

```python
import alquimodelia

# The input arguments
input_args = {
    "X_timeseries": 168,
    "Y_timeseries": 24,
    "n_features_train": 18,
    "n_features_predict": 1,
}
# This is make a model with shapes:
    # input_shape = (N, 168, 18)
    # output_shape = (N, 24, 1)

forearch = alquimodelia.CNNArch(**input_args)

# Now for Vanilla and Stacked CNN:
architecture_args = {}
VanillaCNN = forearch.architecture(**architecture_args)

architecture_args = {"block_repetition": 2}
StackedCNN = forearch.architecture(**architecture_args)

# Keras Models ready to use:
VanillaCNN.summary()
StackedCNN.summary()


```

## [Contribution](CONTRIBUTING.md)

Contributions to Alquimodelia are welcome! If you find any issues or have suggestions for improvement, please feel free to contribute. Make sure to update tests as appropriate and follow the contribution guidelines.

## License

Alquimodelia is licensed under the MIT License, which allows you to use, modify, and distribute the package according to the terms of the license. For more details, please refer to the [LICENSE](LICENSE) file.
