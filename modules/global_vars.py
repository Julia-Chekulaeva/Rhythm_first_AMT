import numpy as np
import librosa

CHROMA_PATH = f'audio_folder/Chroma gamma.mp3'

SR = 22050

# --- Настройка параметров STFT ---
N_FFT = 1024          # Размер окна БПФ (чем больше, тем выше частотное разрешение)
HOP_LENGTH = 512      # Смещение окна (чем меньше, тем выше временное разрешение)
WIN_LENGTH = N_FFT    # Длина окна (обычно равна n_fft)
WINDOW = 'hann'       # Тип окна (Ханна — стандартный выбор)

ONS_HEIGHT = 0.08
ONS_DISTANCE = 2
ONS_PROMINENCE = 0.03
ONS_WLEN = 10

DETREND_WIN = 5

MIN_SECONDS = 0.05

NOTES_COUNT = 88
NOTE_A_MIN = 27.5
NOTE_FREQS = NOTE_A_MIN * 2 ** (np.arange(NOTES_COUNT) / 12)
NOTE_LETTERS = "A A# B C C# D D# E F F# G G#".split(' ')
NOTE_NAMES = [NOTE_LETTERS[i % 12] + str((i + 9) // 12) for i in np.arange(NOTES_COUNT)]

BINS_PER_SEMI = 5
BINS_PER_OCT = 12 * BINS_PER_SEMI
HALF_BINS_PER_SEMI = BINS_PER_SEMI // 2
N_BINS = NOTES_COUNT * BINS_PER_SEMI - HALF_BINS_PER_SEMI

DELTA_PRED, DELTA_POST = 1, 1
FREQS = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
FREQS_CQT = librosa.cqt_frequencies(n_bins=N_BINS, fmin=NOTE_A_MIN, bins_per_octave=BINS_PER_OCT)

F_QUIET = 70 * 2
COEFFS_FOR_NEW_NOTE = np.array([1.1 ** np.arange(-5, 5)]).T
HARMS_COUNT = 10
AMP_COEFF = 0.7

DIM_CLASSES = 2
NOTES_IDXS_MIDI = np.arange(88) + 21

INTERVALS = np.int16(np.round(np.log2([3, 2]) * BINS_PER_OCT))
HARM_COUNT = INTERVALS.shape[0] * 2 + 1

CHROMA_GENERATED, CHROMA_GENERATED_CQT = None, None