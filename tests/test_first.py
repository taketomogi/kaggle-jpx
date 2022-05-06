import pandas as pd
from kaggle_jpx_package import DataFrames


def test_first():
    assert(1) == 1


def test_DataFrames_is_empty():
    dfs = DataFrames()
    assert dfs.is_empty()


def test_DataFrames_is_not_empty():
    dfs = DataFrames()
    dfs['test'] = pd.DataFrame([[1, 2], [3, 4]])
    assert dfs.is_empty() is False


def test_DataFrames_keys():
    dfs = DataFrames()
    dfs['test'] = pd.DataFrame([[1, 2], [3, 4]])
    dfs['test2'] = pd.DataFrame([[1, 2], [3, 4]])
    keys = dfs.keys()
    assert keys == ['test', 'test2']
