import inspect

from alquimodelia.builders import CNN, FCNN, LSTM, Transformer, UNet, AttResUNet, ResUNet
from alquimodelia.builders.base_builder import SequenceBuilder


class ModelMagia:
    registry = dict()

    def __init__(self):
        pass

    def __new__(cls, model_arch, **kwargs):
        # Dynamically create an instance of the specified model class
        model_arch = model_arch.lower()
        num_sequences = 1
        if "vanilla" in model_arch:
            model_arch = model_arch.replace("vanilla", "")
            num_sequences = 1
        if "stacked" in model_arch:
            num_sequences, model_arch = model_arch.split("stacked")
            if len(num_sequences) == 0:
                num_sequences = None
            else:
                num_sequences = int(num_sequences)
        numbers = "".join([word for word in model_arch if word.isdigit()])
        if len(numbers) > 0:
            model_arch = model_arch.replace(numbers, "")
        interpretation_method_name = ""
        if "transformer" in model_arch:
            interpretation_method_name = model_arch.replace("transformer", "")
            model_arch = model_arch.replace(interpretation_method_name, "")              
        model_class = ModelMagia.registry[model_arch]

        instance = super().__new__(model_class)
        # Inspect the __init__ method of the model class to get its parameters
        init_params = inspect.signature(cls.__init__).parameters
        # Separate kwargs based on the parameters expected by the model's __init__
        modelmagia_kwargs = {
            k: v for k, v in kwargs.items() if k in init_params
        }
        model_kwargs = {
            k: v for k, v in kwargs.items() if k not in init_params
        }
        if issubclass(model_class, SequenceBuilder):
            model_kwargs["num_sequences"] = num_sequences
        if len(numbers) > 0:
            interpretation_filters = int(numbers)
            model_kwargs["interpretation_filters"] = interpretation_filters
        if model_arch == "fcnn":
            print("sss")
        if len(interpretation_method_name)>0:
            model_kwargs["interpretation_method"] = ModelMagia.registry[interpretation_method_name]
        for name, method in cls.__dict__.items():
            if "__" in name:
                continue
            if callable(method) and hasattr(instance, name):
                instance.__dict__[name] = method.__get__(instance, cls)

        cls.__init__(instance, **modelmagia_kwargs)
        instance.__init__(**model_kwargs)

        return instance

    @staticmethod
    def register(constructor):
        # TODO: only register if its a BaseModel subclass
        ModelMagia.registry[constructor.__name__.lower()] = constructor


ModelMagia.register(UNet)
ModelMagia.register(ResUNet)
ModelMagia.register(AttResUNet)
ModelMagia.register(Transformer)
ModelMagia.register(CNN)
ModelMagia.register(FCNN)
ModelMagia.register(LSTM)
