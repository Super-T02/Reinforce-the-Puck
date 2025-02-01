import time

import numpy as np


def time_np_random_choice(probs, batch_size):
    start = time.time()
    indices = np.random.choice(len(probs), size=batch_size, p=probs)
    end = time.time()
    return end - start, indices


def time_cdf_searchsorted(probs, batch_size):
    start = time.time()
    cdf = np.cumsum(probs)  # Build cumulative distribution
    random_vals = np.random.rand(batch_size)
    indices = np.searchsorted(cdf, random_vals)
    end = time.time()
    return end - start, indices


def compare_sampling_methods(size=10000, batch_size=512):
    probs = np.random.rand(size)
    probs /= probs.sum()

    # Time np.random.choice
    t_choice, idx = time_np_random_choice(probs, batch_size)

    # Time CDF + searchsorted
    t_cdf, idx = time_cdf_searchsorted(probs, batch_size)

    return t_choice, t_cdf


if __name__ == "__main__":
    t, idx = time_cdf_searchsorted([0.1, 0.2, 0.7], 20)
    print(idx)

    t_choices = []
    t_cdfs = []
    for i in range(20):
        t_choice, t_cdf = compare_sampling_methods()
        t_choices.append(t_choice)
        t_cdfs.append(t_cdf)
    print("Average time for np.random.choice:", np.mean(t_choices))
    print("Average time for CDF + searchsorted:", np.mean(t_cdfs))
