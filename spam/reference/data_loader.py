# data_loader.py
import os

import numpy as np
from nsml.constants import DATASET_PATH


def test_data_loader(root_path):
    """
    Data loader for test data
    :param root_path: root path of test set.

    :return: data type to use in user's infer() function
    """
    # The loader is only an example, and it does not matter which way you load the data.
    loader = np.loadtxt(os.path.join(root_path, 'test', 'test_data'), delimiter=',', dtype=np.float32)
    return loader


def feed_infer(output_file, infer_func):
    """
    This is a function that implements a way to write the user's inference result to the output file.
    :param output_file(str): File path to write output (Be sure to write in this location.)
           infer_func(function): The user's infer function bound to 'nsml.bind()'
    """
    result = infer_func(os.path.join(DATASET_PATH, 'test'))
    result.to_csv(output_file)

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')
