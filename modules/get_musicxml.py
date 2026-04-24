import numpy as np
from sklearn.cluster import KMeans
from music21 import stream, note, chord, meter, clef, layout
from modules.global_vars import NOTES_IDXS_MIDI, DIM_CLASSES


# PIANO NOTES → PIANO NOTES with left and right hands
def modify_piano_notes(piano_notes):

    notes_classified = []
    piano_notes = piano_notes.copy()

    for idx in np.arange(piano_notes.shape[0]):
        notes = NOTES_IDXS_MIDI[piano_notes[idx].astype(bool)] - 21
        if notes.shape[0] == 0:
            print('No notes found at onset number =', idx)
            notes_classified.append(np.array([[], [], [], []]).T)
            continue
        diff_notes = np.zeros_like(notes)
        diff_notes[:-1] = np.diff(notes)
        note_classes = np.zeros((DIM_CLASSES, notes.shape[0]))
        note_classes[0] = notes
        note_classes[1, :diff_notes.argmax()] = 1
        note_classes[2] = notes < (notes.max() + notes.min()) / 2
        note_classes[3, :(notes.shape[0] + 1) // 2] = 1
        notes_classified.append(note_classes.T)

    # Initialize and fit the model
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.concatenate(notes_classified))
    low_note = 0
    high_note = 87
    check_labels = kmeans.predict(np.array([[low_note, 1, 1, 1], [high_note, 0, 0, 0]]))
    assert(check_labels[0] != check_labels[1])
    left_label = check_labels[0]

    for idx in np.arange(piano_notes.shape[0]):

        notes = notes_classified[idx]
        if notes.shape[0] == 0:
            continue
        labels = kmeans.predict(notes)

        for num, note_idx in enumerate(notes.T[0].astype(int)):
            if labels[num] != left_label:
                piano_notes[idx, note_idx] *= 2

    return piano_notes


# PIANO NOTES + onsets + offsets → MusicXML (основная функция)
def piano_notes_to_mxml(piano_notes, onsets, durations, time_signature):

    # 1. Создаем общую партитуру
    score = stream.Score()

    # 2. Партия правой руки (верхний стан)
    right_hand = stream.Part()
    right_hand.id = 'RightHand'
    right_hand.append(clef.TrebleClef()) # Скрипичный ключ
    right_hand.append(meter.TimeSignature(time_signature))

    # 3. Партия левой руки (нижний стан)
    left_hand = stream.Part()
    left_hand.id = 'LeftHand'
    left_hand.append(clef.BassClef()) # Басовый ключ
    left_hand.append(meter.TimeSignature(time_signature))

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

    # 4. Объединяем руки в фортепианную систему (акколаду)
    staff_group = layout.StaffGroup([right_hand, left_hand], name='Piano', abbreviation='Pno.', symbol='brace')
    score.insert(0, staff_group)
    score.insert(0, right_hand)
    score.insert(0, left_hand)

    return score