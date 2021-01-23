import librosa
import numpy as np

import hparams as hp

def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)

def amp_to_db(x):
    return 20. * np.log10(np.maximum(1e-5, x))

def stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return np.clip(normalize(S), 0, 1)
