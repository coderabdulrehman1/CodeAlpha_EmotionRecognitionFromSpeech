"""
src/features.py

Shared feature-extraction code. The notebooks AND the final app use exactly the same preprocessing.
Do not change the settings below after you have extracted features.
"""
import numpy as np
import librosa

# ---- Settings (must match Day 2) ----
SR = 22050          # sampling rate every clip is resampled to
DURATION = 3        # seconds of audio to keep
OFFSET = 0.5        # skip the first 0.5 s (usually silence)
N_MFCC = 40         # number of MFCC coefficients per frame
HOP_LENGTH = 512    # samples between frames
MAX_FRAMES = 130    # fixed number of frames per clip


def mfcc_from_signal(y, sr=SR):
    """Turn a waveform (1D array) into a fixed-size MFCC matrix of shape (N_MFCC, MAX_FRAMES)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    if mfcc.shape[1] < MAX_FRAMES:                       # short clip -> pad with zeros on the right
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
    return mfcc[:, :MAX_FRAMES].astype(np.float32)       # long clip -> crop


def extract_mfcc(path):
    """Load an audio file and return its MFCC matrix of shape (N_MFCC, MAX_FRAMES)."""
    y, _ = librosa.load(path, sr=SR, duration=DURATION, offset=OFFSET)
    return mfcc_from_signal(y)


def cmn(x):
    """Per-clip normalisation (Day 5b): subtract each clip's own mean from MFCC coefficients 1-39.
    Coefficient 0 (loudness) is kept. Padding frames (all zeros) are ignored when computing the mean."""
    valid = np.any(x != 0, axis=0)
    if valid.sum() < 2:
        valid[:] = True
    out = x.copy()
    out[1:, valid] = x[1:, valid] - x[1:, valid].mean(axis=1, keepdims=True)
    return out.astype(np.float32)
