from alquimodelia.builders.base_builder import BaseBuilder
import keras
#TODO: it all
class Transformer(BaseBuilder):

    def model_input_shape(self):
        raise NotImplementedError

    def get_input_layer(self):
        # This is to get the layer to enter the arch, here you can add augmentation or other processing
        return self.input_layer

    def define_model(self):
        # This is the model definition and it must return the output_layer of the architeture. which can be modified further
        # It should use the get_input_layer to fetch the inital layer.
        # it should set the self.last_arch_layer
        raise NotImplementedError

    def define_output_layer(self):
        # This should only deal with the last layer, it can be used to define the classification for instance, or multiple methods to close the model
        # it should use the last_arch_layer
        # it should define self.output_layer
        raise NotImplementedError

    def model_setup(self):
        # Any needed setup before building and conecting the layers
        raise NotImplementedError