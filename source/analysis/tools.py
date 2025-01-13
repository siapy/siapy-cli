import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_signal(signal: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    smoothed_signal = gaussian_filter1d(signal, sigma=sigma)
    return smoothed_signal


def smooth_relevances(relevances: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    return np.apply_along_axis(
        smooth_signal,
        axis=1,
        arr=np.abs(relevances),
        sigma=sigma,
    ).mean(axis=0)
