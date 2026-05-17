import numpy as np
import librosa
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


def get_X_data(diff_cqt_data):
    sum_diff_cqt = diff_cqt_data.sum(axis=1)
    sum_diff_cqt /= np.tile(sum_diff_cqt.max(axis=1), sum_diff_cqt.shape[1]
                            ).reshape(sum_diff_cqt.shape[1], -1).T
    X = sum_diff_cqt
    return X


def get_y_data(piano_notes_data):
    y = np.int16(piano_notes_data.sum(axis=1) > 0)
    return y