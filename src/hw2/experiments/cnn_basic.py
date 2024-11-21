import pytest

from hw2.models.cnn_basic import CNN
from hw2.util import find_max_batch_size


@pytest.mark.hardware_dependent
def test_maximum_batch_size():
    model = CNN(channels=3)
    input_size = (3, 32, 32)
    device = "cuda"
    start_batch_size = 1
    max_batch_size = find_max_batch_size(model, input_size, device, start_batch_size)
    print(f"Maximum batch size: {max_batch_size}")


def test_varying_batch_size_on_training_performance():
    model = CNN(channels=3)
    raise NotImplementedError("This test is not implemented yet.")

def test_varying_learning_rate_on_training_performance():
    model = CNN(channels=3)
    raise NotImplementedError("This test is not implemented yet.")

def test_different_optimizers():
    model = CNN(channels=3)
    raise NotImplementedError("This test is not implemented yet.")

def test_epochs():
    model = CNN(channels=3)
    raise NotImplementedError("This test is not implemented yet.")
