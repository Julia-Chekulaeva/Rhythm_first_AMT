import librosa
import numpy as np
from scipy.signal import find_peaks
from modules.global_vars import SR, HOP_LENGTH, WIN_LENGTH, ONS_HEIGHT, ONS_DISTANCE, ONS_PROMINENCE, ONS_WLEN, MIN_SECONDS
from scipy.spatial import cKDTree


def get_offset_frames(
        y, sr=SR, hop_length=HOP_LENGTH, height=ONS_HEIGHT, distance=ONS_DISTANCE, 
        prominence=ONS_PROMINENCE, wlen=ONS_WLEN, min_seconds=MIN_SECONDS
    ):

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    oenv = oenv / np.max(oenv)
    onset_frames, _ = find_peaks(oenv, height=height, distance=distance, prominence=prominence, wlen=wlen)

    ons_diff = np.diff(librosa.frames_to_time(onset_frames, sr=SR, hop_length=hop_length))
    ons_diff[ons_diff < min_seconds] = ons_diff.max()
    ons_diff_min_idx = np.argmin(ons_diff)

    # Оцениваем темп для каждого момента времени
    tempo_dynamic = librosa.feature.tempo(onset_envelope=oenv, sr=SR, aggregate=None, std_bpm=1.5, hop_length=hop_length)

    # Используем динамический темп для beat_track
    _, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=hop_length, bpm=tempo_dynamic,
                                             units='frames', trim=False)
    
    mult_coeff = max(1, round(60 / ons_diff[ons_diff_min_idx] / tempo_dynamic[onset_frames[ons_diff_min_idx]]))

    # tempo_dynamic2 = librosa.beat.tempo(onset_envelope=oenv, sr=SR, aggregate=None, std_bpm=1.5, hop_length=hop_length,
    #                                     start_bpm=tempo_dynamic[0] * mult_coeff)

    # print(60 / ons_diff[ons_diff_min_idx], tempo_dynamic[onset_frames[ons_diff_min_idx]])

    # print(np.mean(tempo_dynamic), np.mean(tempo_dynamic2))

    # # # Используем динамический темп для beat_track
    # _, beat_frames2 = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=hop_length, bpm=tempo_dynamic2,
    #                                           units='frames', trim=False)

    # print(np.mean(np.diff(beat_frames)), np.mean(np.diff(beat_frames2)))

    # Используем динамический темп для plp
    # plp_pulse = librosa.beat.plp(y=y, sr=SR, onset_envelope=oenv_mod, hop_length=hop_length, win_length=win_length,
    #                              tempo_min=tempo_dynamic.min(), tempo_max=tempo_dynamic.max())
    
    # beat_frames, _ = find_peaks(plp_pulse, height=ONS_HEIGHT, prominence=0.05)

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

    # Определение размера и силы битов
    if size_hint:
        beats_per_bar = int(size_hint.split('/')[0])
    else:
        beat_strength = onset_strength[beats]
        beats_per_bar_values = np.arange(2, 7)
        beats_per_bar = beats_per_bar_values[
            np.argmax([np.corrcoef(beat_strength[:-b], beat_strength[b:])[0, 1] for b in beats_per_bar_values])
        ]
    time_signature = f"{beats_per_bar}/4"

    return idxs, durations, time_signature