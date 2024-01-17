# Note that keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import numpy as np
import pandas as pd


def adjust_to_multiple(varA, varB):
   if varA % varB == 0:
       return varA
   else:
       return round(varA / varB) * varB
   
def stays_if_not_bigger(n1, n2):
    if n2 > n1:
        return n1
    else:
        return n2


def stays_if_not_smaller(n1, n2):
    if n2 < n1:
        return n1
    else:
        return n2


def max_if_not_bigger(n1, n2):
    if n2 > n1:
        return n1
    else:
        return max(n2, n1)


def moving_mean_with_step(prediction, timesteps, skip_step=1):
    if not skip_step:
        skip_step = 1

    for _i in range(timesteps * skip_step):
        prediction = np.vstack([prediction, np.full(timesteps, np.nan)])

    df = pd.DataFrame(prediction)

    shidt_count = 0
    for col in df.columns:
        df[col] = df[col].shift(shidt_count * skip_step)
        shidt_count += 1

    df.dropna(how="all", inplace=True)

    return df.mean(axis=1).to_numpy()


def moving_mean_predictions(prediction, skip_step=1):
    prediction = np.array(prediction)
    if len(prediction.shape) == 2:
        batch, timesteps = prediction.shape
        classes = 1
    elif len(prediction.shape) == 3:  # LETS assume it is ==3
        batch, timesteps, classes = prediction.shape
        if classes == 1:
            prediction = prediction[:, :, 0]

    moving_mean_stack = None
    for _i in range(classes):
        moving_mean = moving_mean_with_step(prediction, timesteps, skip_step)
        if not moving_mean_stack:
            moving_mean_stack = moving_mean
        else:
            moving_mean_stack = np.stack(
                [moving_mean_stack, moving_mean], axis=1
            )
    return moving_mean_stack
