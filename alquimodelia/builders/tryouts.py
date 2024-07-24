import os

os.environ["KERAS_BACKEND"] = "torch"  # @param ["tensorflow", "jax", "torch"]
from alquimodelia.builders.fcnn import FCNN
from alquimodelia.builders.lstm import LSTM

mm = FCNN(
    x_timesteps=168,
    y_timesteps=24,
    num_features_to_train=17,
    num_sequences=None,
)
mm.model.summary()


mm = FCNN(
    x_timesteps=168, y_timesteps=24, num_features_to_train=17, num_sequences=3
)
mm.model.summary()

mm = FCNN(
    x_timesteps=168, y_timesteps=24, num_features_to_train=17, num_sequences=1
)
mm.model.summary()

print("ss")


mm = LSTM(
    x_timesteps=168, y_timesteps=24, num_features_to_train=17, num_sequences=1
)
mm.model.summary()


mm = LSTM(
    x_timesteps=168, y_timesteps=24, num_features_to_train=17, num_sequences=3
)
mm.model.summary()


print("ss")
