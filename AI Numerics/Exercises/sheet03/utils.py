import numpy as np


def ruin_data(data):
    missing_indices = np.random.choice(data.index, size=3, replace=False)
    data.loc[missing_indices, "sepal length (cm)"] = np.nan  # Set some values as NaN

    data.loc[47, "sepal width (cm)"] = 10000

    return data
