import os

os.environ["KERAS_BACKEND"] = "torch"  # @param ["tensorflow", "jax", "torch"]
from alquimodelia.builders.fcnn import FCNN
from alquimodelia.builders.lstm import LSTM
from alquimodelia.builders.resnet import ResNet

# mm = UNet(
#     x_timesteps=16,
#     y_timesteps=1,
#     num_features_to_train=12,
#     num_classes=4,
    # activation_end="relu",

#     x_height= 2000,
#     # x_width= 1,

#     # y_height=1,
#     # y_width=1,
# )
# mm.model.summary()

mm = ResNet(
    x_timesteps=16,
    y_timesteps=1,
    num_features_to_train=12,
    num_classes=4,
    x_height= 32,
    x_width= 32,
    activation_end="relu",
)
mm.model.summary()


mm = ResNet(
    x_timesteps=16,
    y_timesteps=1,
    num_features_to_train=12,
    num_classes=4,
    x_height= 16,
    x_width= 16,
    activation_end="relu",
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
