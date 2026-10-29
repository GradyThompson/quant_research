import numpy as np
import pandas as pd


def test_numpy():
    result = np.sum([1, 2, 3])
    assert result == 6


def test_pandas():
    df = pd.DataFrame({"price": [100, 101, 102]})

    assert len(df) == 3
    assert df["price"].iloc[-1] == 102
