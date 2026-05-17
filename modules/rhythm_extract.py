import librosa
import numpy as np
from scipy.signal import find_peaks
from modules.global_vars import SR, HOP_LENGTH, WIN_LENGTH, ONS_HEIGHT, ONS_DISTANCE, ONS_PROMINENCE, ONS_WLEN, MIN_SECONDS
from modules.global_vars import DELTA_PRED, DELTA_POST, NOTE_A_MIN, N_BINS, BINS_PER_OCT, DETREND_WIN
from scipy.spatial import cKDTree


def get_offset_frames(
        y, sr=SR, hop_length=HOP_LENGTH, height=ONS_HEIGHT, distance=ONS_DISTANCE, 
        prominence=ONS_PROMINENCE, wlen=ONS_WLEN, min_seconds=MIN_SECONDS, detrend_win=DETREND_WIN
    ):

    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=NOTE_A_MIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCT,
                             hop_length=HOP_LENGTH)).T
    cqt = np.concat([np.zeros((1, cqt.shape[1])), cqt])
    diff_cqt = np.maximum(0, np.diff(cqt, axis=0))

    oenv = diff_cqt.sum(axis=1)
    oenv = oenv / np.max(oenv)
    onset_frames, _ = find_peaks(oenv, height=height, distance=distance, prominence=prominence, wlen=wlen)

    ons_diff = np.diff(librosa.frames_to_time(onset_frames, sr=SR, hop_length=hop_length))
    if ons_diff.shape[0] == 0:
        return onset_frames, onset_frames, oenv, 1
    ons_diff[ons_diff < min_seconds] = ons_diff.max()
    ons_diff_min_idx = np.argmin(ons_diff)

    oenv2 = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    tempo_dynamic = librosa.feature.tempo(start_bpm=120, onset_envelope=oenv2, sr=SR, aggregate=None, std_bpm=1.5, hop_length=hop_length)

    _, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=hop_length, bpm=tempo_dynamic,
                                             units='frames', trim=False, tightness=80)
    
    mult_coeff = 4 #max(1, round(60 / ons_diff[ons_diff_min_idx] / tempo_dynamic[onset_frames[ons_diff_min_idx]]))

    return onset_frames, beat_frames, oenv, mult_coeff


def match_onsets_to_beats(onsets, beats, mult_coeff, onset_strength=None, size_hint=None):

    mini_beats = (
        beats[:-1, np.newaxis] @ np.ones((1, mult_coeff)) + 
        (np.arange(0, 1, 1 / mult_coeff)[:, np.newaxis] @ np.diff(beats)[np.newaxis, :]).T
    ).flatten()

    beat_tree = cKDTree(mini_beats[:, np.newaxis])
    dists, idxs = beat_tree.query(onsets[:, np.newaxis], k=1)
    dists = dists.flatten()
    idxs = idxs.flatten()
    
    durations = np.concat([np.diff(idxs), [1]]) / mult_coeff

    if size_hint:
        beats_per_bar = int(size_hint.split('/')[0])
    else:
        beat_strength = onset_strength[beats]
        beats_per_bar_values = np.arange(2, 5)
        corr = [np.corrcoef(beat_strength[:-b], beat_strength[b:])[0, 1] for b in beats_per_bar_values]
        beats_per_bar = beats_per_bar_values[np.argmax(corr)]

        div = 4
        
        if beats_per_bar == 3 and corr[0] > corr[1]:
            div //= 2
            beats_per_bar //= 2
    time_signature = f"{beats_per_bar}/{div}"

    return idxs, durations, time_signature