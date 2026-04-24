import numpy as np
import librosa
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


def generate_chroma_notes(
        chroma_path, sr, hop_length, win_length, window, n_fft, n_bins, note_A_min, bins_per_octave, delta_pred, delta_post, notes_count
    ):
    start_note, end_note = 12, 48
    chroma_audio, _ = librosa.load(chroma_path, sr=sr)
    chroma_D = np.abs(librosa.stft(chroma_audio, n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window))
    chroma_diff_D = np.maximum(0, chroma_D[:, 1:] - chroma_D[:, :-1])
    oenv_chroma = librosa.onset.onset_strength(y=chroma_audio, sr=sr, hop_length=hop_length)
    # Нормализуем для удобства
    oenv_chroma = oenv_chroma / np.max(oenv_chroma)

    # Ищем пики выше 0.08, с минимальным расстоянием 1 кадр
    peaks_chroma, _ = find_peaks(oenv_chroma, height=0.08, distance=1)
    peaks_chroma = peaks_chroma[1:-1]

    start_D, end_D = peaks_chroma[start_note - 1], peaks_chroma[end_note]
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(chroma_diff_D[:, start_D:end_D], sr=sr,  hop_length=hop_length, x_axis='time', y_axis='log')
    chroma_times = librosa.frames_to_time(peaks_chroma[start_note:end_note] - start_D, sr=sr, hop_length=hop_length)
    for t in chroma_times:
        plt.plot([t, t], [0, sr / 2], c='red')
    # plt.scatter(chroma_times, note_freqs[start_note:end_note], color='green', alpha=0.8)
    plt.show()

    chroma_cqt = np.abs(librosa.cqt(chroma_audio, sr=sr, fmin=note_A_min, n_bins=n_bins, bins_per_octave=bins_per_octave,
                                    hop_length=hop_length))
    chroma_diff_cqt = np.maximum(0, chroma_cqt[:, 1:] - chroma_cqt[:, :-1])
    # plt.figure(figsize=(12, 4))
    # librosa.display.specshow(chroma_diff_cqt[:], sr=sr, x_axis='time')
    # plt.show()

    chroma_generated = np.zeros((notes_count, chroma_diff_D.shape[0]))
    chroma_generated_cqt = np.zeros((notes_count, chroma_diff_cqt.shape[0]))

    for i in range(notes_count):
        p = peaks_chroma[i]
        spectrum = chroma_diff_D[:, p - delta_pred : p + delta_post].sum(axis=1)
        spectrum_cqt = chroma_diff_cqt[:, p - delta_pred : p + delta_post].sum(axis=1)
        chroma_generated[i] = spectrum / spectrum.max()
        chroma_generated_cqt[i] = spectrum_cqt / spectrum_cqt.max()
        print(i, chroma_generated_cqt[i].max(), spectrum_cqt.max())
        # f = note_freqs[i]
        # f_low = f / 2 ** (1 / 12)
        # f_high = f * 2 ** (1 / 12)

    
    return chroma_generated, chroma_generated_cqt


def generate_chroma_notes_2(
        harms_count, sr, hop_length, n_bins, note_A_min, bins_per_octave, notes_count, note_freqs, amp_coeff
    ):
    chroma_generated_cqt = []

    for note_idx in range(notes_count):
        chroma_generated_cqt.append([])

        for beta_inharm in np.linspace(0, 0.1, 3):
            
            f = note_freqs[note_idx]
            x_arg = np.arange(sr) * 2 * np.pi / sr * f
            signal = np.zeros(sr)

            signal = np.zeros(sr)
            for i in range(harms_count):
                signal += np.sin(x_arg * i * (1 + beta_inharm * i ** 2)) * amp_coeff ** i

            cqt = np.abs(librosa.cqt(signal, sr=sr, fmin=note_A_min, n_bins=n_bins, bins_per_octave=bins_per_octave,
                                    hop_length=hop_length))
            spectrum_cqt = cqt.sum(axis=1)
            # plt.figure(figsize=(12, 4))
            # librosa.display.specshow(cqt, sr=sr,  hop_length=hop_length, x_axis='time')
            # plt.show()
            chroma_generated_cqt[note_idx].append(spectrum_cqt / spectrum_cqt.max())
        
    chroma_generated_cqt = np.array(chroma_generated_cqt)

    return None, chroma_generated_cqt