import inspect

from alquimodelia.builders import UNet


class ModelMagia:
    registry = dict()

    def __init__(self):
        pass

    def __new__(cls, model_arch, **kwargs):
        # Dynamically create an instance of the specified model class
        model_arch = model_arch.lower()
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

        for name, method in cls.__dict__.items():
            if "__" in name:
                continue
            if callable(method) and hasattr(instance, name):
                instance.__dict__[name] = method.__get__(instance, cls)

        instance.__init__(**model_kwargs)
        cls.__init__(instance, **modelmagia_kwargs)

        return instance

    @staticmethod
    def register(constructor):
        # TODO: only register if its a BaseModel subclass
        ModelMagia.registry[constructor.__name__.lower()] = constructor


ModelMagia.register(UNet)
