from modules.rhythm_extract import get_offset_frames, match_onsets_to_beats
from modules.get_musicxml import modify_piano_notes, piano_notes_to_mxml
import numpy as np
import librosa
from tf_keras import models
from matplotlib import pyplot as plt
from modules.global_vars import *
from modules.help_funcs import get_X_data

# FILE_NAME = 'Mihail_Pletnyov_SHopen_Vals_3_lya_minor_dlya_fortepiano_Op_34_2'
# FILE_NAME = '01-utr-molitva'
# FILE_NAME = '02-zimnee-utro'
# FILE_NAME = '10-polka'
FILE_NAME = 'turkish-march-mozart-rondo-alla-turca'
# FILE_NAME = 'audio_2026-03-10_19-06-14'
# FILE_NAME = '15-italy-pesenka'

PATH_TO_AUDIO = f'sound_examples/{FILE_NAME}.mp3'
PATH_TO_MXML = f'musicxml_folder/{FILE_NAME}.musicxml'

OFFSET, DURATION = 0, 60


def pipeline(y, path_to_mxml=PATH_TO_MXML, display_idx=-1, delay=0, onset_frames_init=None, 
             m_path='vec2vec_model/model_name v7.keras', threshold=0.2):

    window = np.ones(SR * 2 + 1)
    avg_arr_start = np.ones(SR) * y[:SR + 1].mean()
    avg_arr_end = np.ones(SR) * y[-SR - 1:].mean()
    amp = np.convolve(np.concat([avg_arr_start, np.abs(y), avg_arr_end]), window, mode='valid')
    amp /= amp.max()
    amp = np.maximum(0.1, amp)
    y /= amp
    y /= y.max()
    onset_frames, beat_frames, oenv, mult_coeff = get_offset_frames(y, SR, hop_length=HOP_LENGTH)
    if onset_frames_init is not None:
        onset_frames = onset_frames_init
    
    onset_frames -= delay
    cqt = np.abs(librosa.cqt(y, sr=SR, fmin=NOTE_A_MIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCT,
                             hop_length=HOP_LENGTH)).T
    cqt = np.concat([np.zeros((DELTA_PRED, cqt.shape[1])), cqt, np.zeros((DELTA_POST + 1, cqt.shape[1]))])
    diff_cqt = np.maximum(0, np.diff(cqt, axis=0))

    cqt_onsets = np.array([
        diff_cqt[ons : ons + DELTA_PRED + DELTA_POST + 1]
    for ons in onset_frames])

    if cqt_onsets.shape[0] == 0:
        return onset_frames, np.zeros(0)

    for i in range(cqt_onsets.shape[0]):
        if i == display_idx:
            # fig = plt.figure(figsize=(12, 4))
            librosa.display.specshow(cqt_onsets[i], color='blue')
            # fig.axes[0].set_xscale('log')
            plt.show()

    X = get_X_data(cqt_onsets)
    model = models.load_model(m_path)
    all_note_onsets = model.predict(X) > threshold
    idxs_with_notes = all_note_onsets.sum(axis=1) > 0
    print('Notes found at onsets', onset_frames[idxs_with_notes])
    print('No notes found at onsets', onset_frames[np.bool(1 - idxs_with_notes)])
    all_note_onsets = all_note_onsets[idxs_with_notes]
    onset_frames = onset_frames[idxs_with_notes]

    ons_idxs, durations, time_signature = match_onsets_to_beats(
        onset_frames, beat_frames, mult_coeff, onset_strength=oenv, size_hint=None
    )

    idxs_for_0_dur = np.arange(durations.shape[0], dtype=np.int16)
    for i in range(durations.shape[0] - 1, 0, -1):
        if durations[i] > 0:
            for j in range(i - 1, -1):
                if durations[j] > 0:
                    continue
                idxs_for_0_dur[j] = i

    for i in range(idxs_for_0_dur.shape[0]):
        all_note_onsets[idxs_for_0_dur[i]] *= all_note_onsets[i]
    
    piano_notes = modify_piano_notes(all_note_onsets)
    
    idxs_pos_dur = list(set(idxs_for_0_dur))
    score, key = piano_notes_to_mxml(
        piano_notes[idxs_pos_dur], ons_idxs[idxs_pos_dur] / mult_coeff, 
        durations[idxs_pos_dur], time_signature
    )
    score, _ = piano_notes_to_mxml(
        piano_notes[idxs_pos_dur], ons_idxs[idxs_pos_dur] / mult_coeff, 
        durations[idxs_pos_dur], time_signature, key_signature=key
    )

    if path_to_mxml is not None:
        score.write('musicxml', fp=path_to_mxml)
    return onset_frames, piano_notes


if __name__ == '__main__':

    y_init, sr_init = librosa.load(PATH_TO_AUDIO, offset=OFFSET, duration=DURATION)
    y = librosa.resample(y=y_init, orig_sr=sr_init, target_sr=SR)
    onset_frames, piano_notes = pipeline(y, path_to_mxml=PATH_TO_MXML, display_idx=-1)