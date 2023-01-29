import pytest
import numpy as np
import util
import logging

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


@pytest.fixture
def input_data():
    return np.random.rand(100, 32 * 32 * 3)


def test_normalization(input_data):
    func_output = util.normalize_data(input_data)
    for index, row in enumerate(input_data):
        for color in range(3):
            color_avg = np.mean(input_data[index, 32 * 32 * color: 32 * 32 * (color + 1)])
            color_std = np.std(input_data[index, 32 * 32 * color: 32 * 32 * (color + 1)])
            for j, pixel in enumerate(input_data[index, 32 * 32 * color: 32 * 32 * (color + 1)]):
                input_data[index, 32 * 32 * color + j] = (pixel - color_avg) / color_std
    assert np.array_equal(func_output, input_data)


def test_one_hot():
    labels = np.array(['a', 'b', 'c', 'a', 'd', 'b', 'c', 'e', 'f'])
    desired_output = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])
    assert np.array_equal(util.one_hot_encoding(labels, num_classes=6), desired_output)
