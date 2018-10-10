from margipose.data.mixed import RoundRobinSampler
import numpy as np


def test_round_robin_sampler():
    sampler = RoundRobinSampler([[1, 3, 5], [2, 4, 6]], 6)
    indices = np.array([i for i in sampler])

    # Examples should alternate between the two sources
    np.testing.assert_array_equal(indices % 2, np.array([1, 0, 1, 0, 1, 0]))
    # All examples should be represented (sampling without replacement)
    np.testing.assert_array_equal(np.sort(indices), np.array([1, 2, 3, 4, 5, 6]))
