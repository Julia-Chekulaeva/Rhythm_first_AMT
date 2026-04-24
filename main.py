from modules.notes_extract import get_onsets_pianoroll_2
from modules.rythm_extract import get_offset_frames, match_onsets_to_beats
from modules.get_musicxml import modify_piano_notes, piano_notes_to_mxml
import numpy as np
import librosa
from modules.global_vars import SR, HOP_LENGTH, NOTE_A_MIN, N_BINS, BINS_PER_OCT

# FILE_NAME = 'Mihail_Pletnyov_SHopen_Vals_3_lya_minor_dlya_fortepiano_Op_34_2'
# FILE_NAME = '01-utr-molitva'
# FILE_NAME = '02-zimnee-utro'
# FILE_NAME = '10-polka'
FILE_NAME = 'turkish-march-mozart-rondo-alla-turca'
# FILE_NAME = 'Example_diploma'
# FILE_NAME = '15-italy-pesenka'

PATH_TO_AUDIO = f'sound_examples/{FILE_NAME}.mp3'
PATH_TO_MXML = f'musicxml_folder/{FILE_NAME}.musicxml'

offset, duration = 0, 20


if __name__ == '__main__':

    y_init, sr_init = librosa.load(PATH_TO_AUDIO, offset=offset, duration=duration, sr=SR)
    y = librosa.resample(y=y_init, orig_sr=sr_init, target_sr=SR)
    onset_frames, beat_frames, oenv, mult_coeff = get_offset_frames(y, SR, hop_length=HOP_LENGTH)
    ons_idxs, durations, time_signature = match_onsets_to_beats(
        onset_frames, beat_frames, mult_coeff, onset_strength=oenv, size_hint=None
    )
    cqt = np.abs(librosa.cqt(y, sr=SR, fmin=NOTE_A_MIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCT,
                             hop_length=HOP_LENGTH))
    _, all_note_onsets = get_onsets_pianoroll_2(cqt=cqt, onset_frames=onset_frames, display_idx=1)
    piano_notes = modify_piano_notes(all_note_onsets)
    score = piano_notes_to_mxml(piano_notes, ons_idxs / mult_coeff, durations, time_signature)

    score.write('musicxml', fp=PATH_TO_MXML)
