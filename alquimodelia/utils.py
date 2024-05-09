# Note that keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import numpy as np
from keras import ops


def adjust_to_multiple(varA, varB):
    if varA % varB == 0:
        return varA
    else:
        return round(varA / varB) * varB


def max_if_not_bigger(n1, n2):
    if n2 > n1:
        return n1
    else:
        return max(n2, n1)


def repeat_elem(tensor, rep, dimension_to_repeat=-1):
    tile_shape = np.ones(len(tensor.shape) - 1)
    tile_shape[dimension_to_repeat] = rep
    return ops.tile(tensor, tile_shape)


def count_number_divisions(size: int, count: int, by: int = 2, limit: int = 2):
    """
    Count the number of possible steps.

    Parameters
    ----------
    size : int
        Image size (considering it is a square).
    count : int
        Input must be 0.
    by : int, optional
        The factor by which the size is divided. Default is 2.
    limit : int, optional
        Size of last filter (smaller). Default is 2.

    Returns
    -------
    int
        The number of possible steps.
    """
    if size >= limit:
        if size % 2 == 0:
            count = count_number_divisions(
                size / by, count + 1, by=by, limit=limit
            )
    else:
        count = count - 1
    return count
