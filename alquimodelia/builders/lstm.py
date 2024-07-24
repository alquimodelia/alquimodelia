import keras

from alquimodelia.builders.base_builder import SequenceBuilder

# TODO: build the other models in https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/


class LSTM(SequenceBuilder):
    """Base classe for LSTM models."""

    def __init__(
        self,
        num_sequences: int = 1,
        flatten_output: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_sequences=num_sequences,
            flatten_output=flatten_output,
            **kwargs,
        )

    def model_setup(self):
        if self.num_sequences > 1:
            self.sequence_args = [{"initial_block": True}]
            for i in range(self.num_sequences - 2):
                self.sequence_args.append({"initial_block": True})
            self.sequence_args.append({})

    def arch_block(
        self,
        x,
        lstm_args=None,
        filter_enlarger=4,
        filter_limit=200,
        initial_block=False,
    ):
        """Defines the architecture block for the LSTM layer.

        This method defines a block of operations that includes an LSTM layer. The arguments for the LSTM layer are determined by the `lstm_args` parameter.

        Parameters:
        -----------
        x: keras.layer
            Input layer
        lstm_args: dict
            Arguments for the LSTM layer. Default is {"units": 50, "activation": "relu"}.
        filter_enlarger: int
            Multiplier for the number of filters in the dense layers. Default is 4.
        filter_limit: int
            Maximum number of filters in the dense layers. Default is 200.

        Returns:
        --------
        x: keras.layer
            Output layer
        """
        # TODO: set the units by the filters like conv
        # Default LSTM arguments
        units = self.filters or 50
        if lstm_args is None:
            lstm_args = {"units": units, "activation": "relu"}
        default_lstm_args = {"units": units, "activation": "relu"}

        # If lstm_args is provided, update the default arguments
        if lstm_args is not None:
            default_lstm_args.update(lstm_args)
        if initial_block:
            default_lstm_args.update({"return_sequences": True})

        # Apply LSTM layer
        x = keras.layers.LSTM(**default_lstm_args)(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        return x
