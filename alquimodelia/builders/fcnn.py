from functools import cached_property

import numpy as np

from alquimodelia.builders.base_builder import SequenceBuilder

# TODO: recheck everything about filters and stuffy
# TODO: make a thingy for s staircase of filter numbers


class FCNN(SequenceBuilder):
    """Base classe for FCNN models."""

    def __init__(
        self,
        num_sequences: int = 1,
        flatten_input: bool = True,
        flatten_output: bool = True,
        division_base_power: int = 2,
        division_space_method="max",  # min mean
        division_per_dim=True,
        cutoff_filters: bool = True,
        **kwargs,
    ):
        if num_sequences is not None:
            num_sequences -= 1
        self.division_base_power = division_base_power
        self.log_div_filters = []
        self.division_per_dim = division_per_dim
        self.division_space_method = getattr(np, division_space_method)
        self.cutoff_filters = cutoff_filters
        super().__init__(
            num_sequences=num_sequences,
            flatten_input=flatten_input,
            flatten_output=flatten_output,
            **kwargs,
        )

    @cached_property
    def number_of_log_divisions(self):
        if self.num_sequences is None:
            dimension_on_input = np.prod(self.model_input_shape)
            dimension_on_output = np.prod(self.model_output_shape)
            division_count = -1
            division_value = dimension_on_input
            while division_value > dimension_on_output:
                division_value /= self.division_base_power
                division_count += 1
            num_dense_layers = division_count
            if num_dense_layers % 2 != 0:
                num_dense_layers -= 1
            self.num_sequences = int(num_dense_layers / 2)
        else:
            num_dense_layers = self.num_sequences * 2

        return num_dense_layers

    @cached_property
    def number_division(self):
        if self.num_sequences is None:
            divisions_in_dim = []
            for dim_name in ["T", "W", "H", "B"]:
                input_dim = self.dict_dimension_to_use.get(dim_name, 1)
                output_dim = self.dict_dimension_to_predict.get(dim_name, 1)
                division_count = -1
                division_value = input_dim
                while division_value > output_dim:
                    division_value /= self.division_base_power
                    division_count += 1

                if division_count <= 1:
                    continue
                divisions_in_dim.append(division_count)
            num_div = self.division_space_method(divisions_in_dim)
            if num_div % 2 != 0:
                num_div -= 1
            self.num_sequences = int(num_div / 2) - 1
        else:
            num_div = (self.num_sequences + 1) * 2

        return int(num_div)

    def model_setup(self):
        self.arch_block = self.dense_block
        dimension_on_output = np.prod(self.model_output_shape)
        dimension_on_input = np.prod(self.model_input_shape)

        self.sequence_args = []  # TODO: do not overwirte
        # TODO: bigger than  needed
        last_log_value = dimension_on_output
        if self.division_per_dim:
            if self.number_division > 2:
                for i in range(0, (self.number_division - 2), 2):
                    self.sequence_args.append(dimension_on_output* (self.division_base_power ** (i + 2)))
                    self.sequence_args.append(dimension_on_output* (self.division_base_power ** (i + 3)))

                self.sequence_args.reverse()
                if self.cutoff_filters or self.filters is not None:
                    cutoff_filters = self.filters or dimension_on_input
                    if max(self.sequence_args)>dimension_on_input:

                        max_sequence = min(cutoff_filters, max(self.sequence_args))
                        min_sequence = max(dimension_on_output, min(self.sequence_args))
                        # This is repeated so we have what would be the second after the dimension in
                        sequence_args = min_sequence+(((self.sequence_args-min(self.sequence_args))*(max_sequence-min_sequence))/(max(self.sequence_args)-min(self.sequence_args)))
                        sequence_args = sequence_args.round().astype(int)
                        if self.filters is None:
                            max_sequence = sequence_args[1]
                            self.sequence_args = min_sequence+(((self.sequence_args-min(self.sequence_args))*(max_sequence-min_sequence))/(max(self.sequence_args)-min(self.sequence_args)))
                            self.sequence_args = self.sequence_args.round().astype(int)
                        else:
                            self.sequence_args = sequence_args

                sequence_args = []
                for i in range(0, len(self.sequence_args), 2):
                    sequence_args.append({
                        "dense_args": [
                            {
                                "units": self.sequence_args[i]
                            },
                            {
                                "units": self.sequence_args[i+1]
                            },
                        ]
                    })
                self.sequence_args=sequence_args
            last_log_value = dimension_on_output * (self.division_base_power)
            if self.num_sequences < 1:
                last_log_value = min(200, last_log_value)
            last_log_value = self.interpretation_filters or last_log_value
            self.interpretation_dense_args = [
                {"units": int(last_log_value)},
                {"units": dimension_on_output},
            ]

        else:
            if self.number_of_log_divisions > 0:
                sequence_args = []
                for i in range(0, self.number_of_log_divisions, 2):
                    sequence_args.append(
                        dimension_on_output
                        * (self.division_base_power ** (i + 2))
                    )
                    sequence_args.append(
                        dimension_on_output
                        * (self.division_base_power ** (i + 3))
                    )
                sequence_args = np.array(sequence_args)
                if self.filters is not None:
                    min_val= np.min(sequence_args)
                    max_val = np.max(sequence_args)
                    new_max = self.filters
                    sequence_args = min_val + ((sequence_args - min_val) * (new_max - min_val)) / (max_val - min_val)
                    sequence_args = sequence_args.round().astype(int)
                for i in range(0, self.number_of_log_divisions, 2):
                    self.sequence_args.append(
                        {
                            "dense_args": [
                                {
                                    "units": sequence_args[i + 1]
                                },
                                {
                                    "units": sequence_args[i]
                                },
                            ]
                        }
                    )
                self.sequence_args.reverse()
            last_log_value = dimension_on_output * (
                self.division_base_power ** (1)
            )
            value_to_use = int((last_log_value - dimension_on_output) / 2)
            value_to_use = max(value_to_use, 1)
            if self.num_sequences < 1:
                value_to_use = min(200, value_to_use)
            value_to_use = self.interpretation_filters or value_to_use
            self.interpretation_dense_args = [
                {"units": value_to_use},
                {"units": dimension_on_output},
            ]
        
