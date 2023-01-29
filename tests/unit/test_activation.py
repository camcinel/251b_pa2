import neuralnet as net
import numpy as np
import logging
import pytest

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


@pytest.fixture
def test_arr():
    return np.random.rand(200)


def test_sigmoid(test_arr):
    sigmoid = net.Activation(activation_type='sigmoid')
    desired_output = np.zeros(test_arr.shape)
    for index, entry in enumerate(test_arr):
        desired_output[index] = 1 / (1 + np.exp(-entry))
    assert np.array_equal(sigmoid.sigmoid(test_arr), desired_output)


def test_tanh(test_arr):
    tanh = net.Activation(activation_type='tanh')
    desired_output = np.zeros(test_arr.shape)
    for index, entry in enumerate(test_arr):
        desired_output[index] = np.tanh(entry)
    assert np.array_equal(tanh.tanh(test_arr), desired_output)


def test_relu(test_arr):
    relu = net.Activation(activation_type='ReLU')
    desired_output = np.zeros(test_arr.shape)
    for index, entry in enumerate(test_arr):
        desired_output[index] = max(0, entry)
    assert np.array_equal(relu.ReLU(test_arr), desired_output)


def test_softmax(test_arr):
    softmax = net.Activation(activation_type='output')
    test_arr = test_arr.reshape(10, 20)
    desired_output = np.zeros(test_arr.shape)
    for i, row in enumerate(test_arr):
        row_sum = np.sum(np.exp(row))
        for j, column in enumerate(row):
            desired_output[i, j] = np.exp(column) / row_sum
    assert np.array_equal(softmax.output(test_arr), desired_output)
