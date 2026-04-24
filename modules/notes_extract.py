import librosa
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from modules.global_vars import BINS_PER_SEMI, NOTE_A_MIN, SR, HOP_LENGTH, DELTA_PRED, DELTA_POST, NOTE_FREQS, COEFFS_FOR_NEW_NOTE, NOTES_COUNT
from modules.global_vars import CHROMA_GENERATED_CQT, FREQS_CQT, N_BINS


def get_onsets_pianoroll(
        cqt, onset_frames, freqs_cqt=FREQS_CQT, sr=SR, hop_length=HOP_LENGTH, delta_pred=DELTA_PRED, delta_post=DELTA_POST
    ):
    all_notes = -np.ones((cqt.shape[1], NOTES_COUNT))
    all_note_onsets = np.zeros((onset_frames.shape[0], NOTES_COUNT))
    notes_to_check = np.zeros(NOTES_COUNT, dtype=np.bool_)
    notes_range = np.arange(NOTES_COUNT)
    stop_idx = 1
    for ons_idx, onset_moment in enumerate(onset_frames):
        offset_moment = (onset_frames[ons_idx + 1] if ons_idx < onset_frames.shape[0] - 1 else 
                         onset_moment + delta_post) - 1
        delta_post_corrected = min(delta_post, offset_moment - onset_moment - delta_pred)
        current_cqt = cqt[:, onset_moment - delta_pred : onset_moment + delta_post_corrected]
        # S_old = min(current_S[1:], current_S[:-1])
        # S_new = 
        diff_cqt = np.maximum(0, current_cqt[:, 1:] - current_cqt[:, :-1])
        sum_diff_cqt = np.sum(diff_cqt, axis=1)
        # sum_onset_cqt = np.mean(cqt[:, onset_moment : onset_moment + delta_post_corrected], axis=1)
        spectrum_cqt = np.mean([sum_diff_cqt, cqt[:, onset_moment + delta_post_corrected]], axis=0)
        # spectrum_cqt = cqt[:, onset_moment + delta_post_corrected]
        max_spectrum = np.max(spectrum_cqt)
        spectrum_cqt /= max_spectrum
        peaks_indices, _ = find_peaks(
            spectrum_cqt,
            height=0.05,                    # 5% от глобального максимума
            prominence=0.1,                 # Минимальная "заметность" пика
            distance=BINS_PER_SEMI - 1,    # Минимальное расстояние между пиками в бинах
            wlen=BINS_PER_SEMI * 3         # Размер окна для расчёта prominence
        )
        if ons_idx == stop_idx:
            librosa.display.specshow(current_cqt, sr=sr, hop_length=hop_length)
            plt.show()

        spectrum_cqt_freqs = freqs_cqt[peaks_indices]
        if spectrum_cqt_freqs.shape[0] == 0:
            continue
        spectrum_cqt_freqs = list(spectrum_cqt_freqs)
        
        f0 = min(spectrum_cqt_freqs[0] * 2, NOTE_A_MIN)
        for i, f in enumerate(spectrum_cqt_freqs.copy()):
            if f > f0:
                break
            spectrum_cqt_freqs.insert(i, f / 2) 

        note_low = 0
        step_back = 1
        notes_to_check_count = 2
        notes_to_check[:] = False
            
        for spectrum_freq in spectrum_cqt_freqs:
            for i in range(note_low, NOTES_COUNT - step_back):
                if NOTE_FREQS[i + step_back] >= spectrum_freq:
                    notes_neighbours = range(i, i + notes_to_check_count)
                    notes_to_check[notes_neighbours] = True
                    note_low = i + notes_to_check_count
                    break

        notes_to_check_list = notes_range[notes_to_check]
        corr_coeffs = np.corrcoef(np.diff(CHROMA_GENERATED_CQT[notes_to_check]), np.diff(spectrum_cqt))[-1, :-1]
        idxs_sorted = corr_coeffs.argsort()[::-1]
        idx0 = idxs_sorted[0]
        note0 = notes_to_check_list[idx0]
        prev_accord_spectrum = CHROMA_GENERATED_CQT[note0 : note0 + 1]
        prev_corr = corr_coeffs[idx0]
        current_notes = [note0]
        print(f'idx = {ons_idx}, onset = {onset_moment}')
        print(f'start_corr = {prev_corr}')
        # prev_corr = 0
        # current_notes = []
        current_spectrum = spectrum_cqt
        
        for i in idxs_sorted[1:]:
        # for i in idxs_sorted:
            note = notes_to_check_list[i]
            note_spectrum = CHROMA_GENERATED_CQT[note]
            if ons_idx == stop_idx:
                fig = plt.figure(figsize=(12, 4))
                plt.plot(freqs_cqt, note_spectrum, color='blue')
                plt.plot(freqs_cqt, current_spectrum, color='green')
                plt.vlines(spectrum_cqt_freqs, ymin=0, ymax=np.max([note_spectrum, current_spectrum]), colors='red')
                fig.axes[0].set_xscale('log')
                plt.show()
            # new_corr = np.corrcoef(note_spectrum, current_spectrum)[0, 1]
            # mask = np.minimum(1, note_spectrum / (current_spectrum + mask_eps))
            # current_spectrum = current_spectrum * (1 - mask)
            new_accord_spectrums = note_spectrum + COEFFS_FOR_NEW_NOTE @ prev_accord_spectrum
            corr_m = np.corrcoef(np.diff(new_accord_spectrums), np.diff(current_spectrum))[:-1, -1]
            max_corr_idx = corr_m.argmax()
            new_corr = corr_m[max_corr_idx]
            if new_corr - prev_corr <= 0.01:
                continue
                # continue
            prev_accord_spectrum[:1] = new_accord_spectrums[max_corr_idx]
            prev_corr = new_corr
            current_notes.append(note)
        
        print(f'corr_current = {prev_corr}, corr_last_calculated = {new_corr}')
        print(current_notes)

        all_notes[onset_moment : offset_moment, current_notes] = 0
        all_notes[onset_moment, current_notes] = 1
        all_note_onsets[ons_idx, current_notes] = 1
    
    return all_notes, all_note_onsets


def get_onsets_pianoroll_2(
        cqt, onset_frames, freqs_cqt=FREQS_CQT, sr=SR, hop_length=HOP_LENGTH, delta_pred=DELTA_PRED, 
        mask_eps=0.001, threshold=0.1, peak_height=0.15, display_idx=-1, 
    ):

    diff_cqt_full = np.concat([np.zeros((N_BINS, 1)), np.diff(cqt)], axis=1)
    window = BINS_PER_SEMI
    half_bins_per_semi = BINS_PER_SEMI // 2
    weights = np.ones(window)
    # weights2 = np.ones(window + 2)
    all_notes = -np.ones((cqt.shape[1], NOTES_COUNT))
    all_note_onsets = np.zeros((onset_frames.shape[0], NOTES_COUNT))
    notes_to_check = np.zeros(NOTES_COUNT, dtype=np.bool_)
    notes_range = np.arange(NOTES_COUNT)
    display_idx = display_idx
    
    for ons_idx, onset_moment in enumerate(onset_frames):
        current_cqt = cqt[:, onset_moment - delta_pred : onset_moment + 1]
        diff_cqt = np.maximum(0, diff_cqt_full[:, onset_moment - delta_pred : onset_moment + 1])
        sum_diff_cqt = np.sum(diff_cqt, axis=1)
        spectrum_cqt = cqt[:, onset_moment]
        mask_from_diff_cqt = np.minimum(1, spectrum_cqt / (sum_diff_cqt + mask_eps))
        mask_prev_spectrum = np.convolve(cqt[:, onset_moment - delta_pred], weights, mode='same'
                                         ) / (spectrum_cqt + mask_eps)
        mask_prev_spectrum = np.minimum(1, mask_prev_spectrum)
        spectrum_cqt = spectrum_cqt * mask_from_diff_cqt * (1 - mask_prev_spectrum)
        max_spectrum = np.max(spectrum_cqt)
        spectrum_cqt /= max_spectrum
        peaks_indices, _ = find_peaks(
            spectrum_cqt[:NOTES_COUNT * BINS_PER_SEMI],
            height=peak_height,
            prominence=0.1,                 # Минимальная "заметность" пика
            distance=BINS_PER_SEMI - 1,     # Минимальное расстояние между пиками в бинах
            wlen=BINS_PER_SEMI * 3          # Размер окна для расчёта prominence
        )
        if ons_idx == display_idx:
            librosa.display.specshow(current_cqt, sr=sr, hop_length=hop_length)
            plt.show()

        spectrum_cqt_freqs = freqs_cqt[peaks_indices]
        if spectrum_cqt_freqs.shape[0] == 0:
            continue
        spectrum_cqt_freqs = list(spectrum_cqt_freqs)

        notes_to_check[(peaks_indices + half_bins_per_semi) // BINS_PER_SEMI] = True
        
        # f0 = min(spectrum_cqt_freqs[0] * 2, NOTE_A_MIN)
        # for i, f in enumerate(spectrum_cqt_freqs.copy()):
        #     if f > f0:
        #         break
        #     spectrum_cqt_freqs.insert(i, f / 2) 

        # note_low = 0
        # step_back = 1
        # notes_to_check_count = 2
        # notes_to_check[:] = False
            
        # for spectrum_freq in spectrum_cqt_freqs:
        #     notes_to_check[]
        #     for i in range(note_low, NOTES_COUNT - step_back):
        #         if NOTE_FREQS[i + step_back] >= spectrum_freq:
        #             notes_neighbours = range(i, i + notes_to_check_count)
        #             notes_to_check[notes_neighbours] = True
        #             note_low = i + notes_to_check_count
        #             break

        notes_to_check_list = notes_range[notes_to_check]
        current_notes = []
        print(f'idx = {ons_idx}, onset = {onset_moment}')
        current_spectrum = np.convolve(spectrum_cqt, weights, mode='same')
        current_spectrum /= current_spectrum.max()
        
        for note in notes_to_check_list:
            freq_right = note * BINS_PER_SEMI + half_bins_per_semi
            freq_left = freq_right - BINS_PER_SEMI
            current_peak_height = current_spectrum[freq_left : freq_right].max()
            if current_peak_height < peak_height:
                continue
            note_spectrum = CHROMA_GENERATED_CQT[note] * current_peak_height
            min_div = np.min((current_spectrum + mask_eps) / (note_spectrum + mask_eps), axis=1)
            max_idx = np.argmax(min_div)
            if min_div[max_idx] < threshold:
                continue
            note_spectrum_convolved = np.convolve(note_spectrum[max_idx], weights, mode='same')
            note_spectrum_convolved /= note_spectrum_convolved.max()
            mask_note = note_spectrum_convolved / (current_spectrum + mask_eps)
            mask_note /= mask_note[freq_left : freq_right].max()
            mask_note = np.minimum(1, mask_note)
            current_spectrum *= (1 - mask_note)
            current_notes.append(note)
            if ons_idx == display_idx:
                fig = plt.figure(figsize=(12, 4))
                plt.plot(freqs_cqt, note_spectrum[max_idx], color='blue')
                plt.plot(freqs_cqt, (mask_note), color='green')
                plt.plot(freqs_cqt, current_spectrum, color='magenta')
                plt.scatter(freqs_cqt, note_spectrum[max_idx], color='blue', s=5)
                plt.scatter(freqs_cqt, mask_note, color='green', s=5)
                plt.scatter(freqs_cqt, current_spectrum, color='magenta', s=5)
                plt.vlines(spectrum_cqt_freqs, ymin=0, ymax=1, colors='red')
                fig.axes[0].set_xscale('log')
                plt.title(f'min_div = {min_div}')
                plt.show()
        
        print(current_notes)

        # for note in current_notes:
        #     diff2_note = np.sum(-diff2_cqt_full[note * BINS_PER_SEMI - 1 : note * BINS_PER_SEMI + 1, onset_moment:], axis=1)
        #     down_frames = frames_range[diff2_note >= threshold]
        #     offset_moment = down_frames[0]
        #     all_notes[onset_moment : offset_moment, current_notes] = 0
        all_notes[onset_moment, current_notes] = 1
        all_note_onsets[ons_idx, current_notes] = 1
    
    return all_notes, all_note_onsets