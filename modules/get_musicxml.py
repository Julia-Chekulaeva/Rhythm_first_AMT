import numpy as np
from sklearn.cluster import KMeans
from music21 import stream, note, chord, meter, clef, layout
from modules.global_vars import NOTES_IDXS_MIDI, NOTE_A_MIN


def modify_piano_notes(piano_notes):

    mid_note_c = 3 + 12 * 3
    piano_notes = np.int16(piano_notes)
    piano_notes[:, mid_note_c:] *= 2

    return piano_notes


def piano_notes_to_mxml(piano_notes, onsets, durations, time_signature, key_signature=None):

    score = stream.Score()

    right_hand = stream.Part()
    right_hand.id = 'RightHand'
    right_hand.append(clef.TrebleClef())
    right_hand.append(meter.TimeSignature(time_signature))

    idxs_pos = piano_notes.sum(axis=1) > 0
    piano_notes = piano_notes[idxs_pos]
    onsets = onsets[idxs_pos]
    durations = np.diff(np.concat([onsets, [onsets[-1] + 1]]))

    left_hand = stream.Part()
    left_hand.id = 'LeftHand'
    left_hand.append(clef.BassClef())
    left_hand.append(meter.TimeSignature(time_signature))

    if key_signature is not None:
        right_hand.append(key_signature)
        left_hand.append(key_signature)

    left_hand.append(note.Rest(quarterLength=onsets[0]))
    right_hand.append(note.Rest(quarterLength=onsets[0]))

    for dur, idx in zip(durations, np.arange(onsets.shape[0])):

        notes_left = [
            note.Note(n) for n in NOTES_IDXS_MIDI[piano_notes[idx] == 1]
        ]
        notes_right = [
            note.Note(n) for n in NOTES_IDXS_MIDI[piano_notes[idx] == 2]
        ]

        if len(notes_left) > 0:
            left_hand.append(chord.Chord(notes=notes_left, quarterLength=dur))
        else:
            left_hand.append(note.Rest(quarterLength=dur))

        if len(notes_right) > 0:
            right_hand.append(chord.Chord(notes=notes_right, quarterLength=dur))
        else:
            right_hand.append(note.Rest(quarterLength=dur))

    staff_group = layout.StaffGroup([right_hand, left_hand], name='Piano', abbreviation='Pno.', symbol='brace')
    score.insert(0, staff_group)
    score.insert(0, right_hand)
    score.insert(0, left_hand)

    key = score.analyze('key')
    print(key.tonic.name, key.mode)

    return score, key